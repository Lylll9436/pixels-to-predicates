#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双层城市感知模型 (Dual-Layer Urban Perception Model)

本模块实现 CEUS 论文的双层框架：
1. Micro-Level: 场景图 + 文本描述的融合特征 (Z_scene)
2. Macro-Level: 城市街道网络上的 GAT 传播
3. Fusion: 微观-宏观特征的门控注意力融合

支持两种微观特征输入模式：
- 模式 A: 直接传入预融合的 micro_embeddings（向后兼容）
- 模式 B: 传入 h_visual + h_caption，内部使用 MicroFusionEncoder 融合

作者: CEUS Project
日期: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    SAGEConv,
    GCNConv,
    TransformerConv,
    GINConv,
)
from torch_scatter import scatter_mean
from typing import Optional, Literal

# 导入微观融合模块
from micro_encoder import MicroFusionEncoder, GatedAttentionFusion


# ==============================================================================
# 微观-宏观门控融合层
# ==============================================================================
class MicroMacroFusion(nn.Module):
    """
    微观-宏观特征门控融合层

    实现公式:
        α = sigmoid(W · [H_micro || H_macro])
        H_final = α * H_micro_aligned + (1 - α) * H_macro
    """

    def __init__(
        self, micro_dim: int, macro_dim: int, hidden_dim: int = 64, dropout: float = 0.2
    ):
        super().__init__()

        self.micro_dim = micro_dim
        self.macro_dim = macro_dim

        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(micro_dim + macro_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # 微观特征对齐（如果维度不同）
        if micro_dim != macro_dim:
            self.micro_align = nn.Linear(micro_dim, macro_dim)
        else:
            self.micro_align = nn.Identity()

        self.output_dim = macro_dim

    def forward(self, h_micro: torch.Tensor, h_macro: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            h_micro: 微观特征 [batch_size, micro_dim]
            h_macro: 宏观特征 [batch_size, macro_dim]

        Returns:
            h_fused: 融合特征 [batch_size, macro_dim]
        """
        combined = torch.cat([h_micro, h_macro], dim=-1)
        alpha = self.gate_network(combined)

        h_micro_aligned = self.micro_align(h_micro)
        h_fused = alpha * h_micro_aligned + (1 - alpha) * h_macro

        return h_fused


# ==============================================================================
# 宏观层 GNN 模块
# ==============================================================================
class MacroGNN(nn.Module):
    """
    宏观层图神经网络

    在城市街道网络上传播特征，整合物理拓扑边和功能相似性边。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GAT 层
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
            self.gnn_layers.append(
                GATConv(
                    in_channels,
                    hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )

        self.output_dim = hidden_dim * num_heads

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]

        Returns:
            h: 传播后的节点特征 [num_nodes, output_dim]
        """
        h = self.input_proj(x)

        for conv in self.gnn_layers:
            h = conv(h, edge_index)
            h = F.elu(h)

        return h


# ==============================================================================
# 可配置宏观层 GNN 模块
# ==============================================================================
class ConfigurableMacroGNN(nn.Module):
    """可配置的宏观层图神经网络，支持多种 GNN 架构"""

    ATTENTION_BASED = {"GAT", "GATv2", "Transformer"}
    VALID_TYPES = {"GAT", "GATv2", "SAGE", "GCN", "Transformer", "GIN"}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        gnn_type: str = "GAT",
        gnn_aggr: str = "mean",
    ):
        super().__init__()

        if gnn_type not in self.VALID_TYPES:
            raise ValueError(f"Unknown GNN type: {gnn_type}. Valid: {self.VALID_TYPES}")

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads if gnn_type in self.ATTENTION_BASED else 1
        self.dropout_rate = dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_ch = hidden_dim if i == 0 else self._get_layer_out_dim(hidden_dim)
            layer = self._create_gnn_layer(in_ch, hidden_dim, dropout, gnn_aggr)
            self.gnn_layers.append(layer)
            self.norms.append(nn.LayerNorm(self._get_layer_out_dim(hidden_dim)))

        self.output_dim = self._get_layer_out_dim(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _get_layer_out_dim(self, hidden_dim: int) -> int:
        if self.gnn_type in self.ATTENTION_BASED:
            return hidden_dim * self.num_heads
        return hidden_dim

    def _create_gnn_layer(
        self, in_channels: int, out_channels: int, dropout: float, aggr: str
    ):
        if self.gnn_type == "GAT":
            return GATConv(
                in_channels,
                out_channels,
                heads=self.num_heads,
                dropout=dropout,
                concat=True,
            )
        elif self.gnn_type == "GATv2":
            return GATv2Conv(
                in_channels,
                out_channels,
                heads=self.num_heads,
                dropout=dropout,
                concat=True,
            )
        elif self.gnn_type == "SAGE":
            return SAGEConv(in_channels, out_channels, aggr=aggr)
        elif self.gnn_type == "GCN":
            return GCNConv(in_channels, out_channels)
        elif self.gnn_type == "Transformer":
            return TransformerConv(
                in_channels,
                out_channels,
                heads=self.num_heads,
                dropout=dropout,
                concat=True,
            )
        elif self.gnn_type == "GIN":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(mlp)
        else:
            raise ValueError(f"Unknown GNN type: {self.gnn_type}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.gnn_layers, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.elu(h)
            h = self.dropout(h)
        return h


# ==============================================================================
# 核心模型：双层感知模型
# ==============================================================================
class DualLayerPerceptionModel(nn.Module):
    """
    双层城市感知模型 (CEUS)

    整合：
    1. Micro-Level: 预计算的图像/场景图嵌入（可选内部融合 visual + caption）
    2. Macro-Level: 城市街道网络上的 GAT 传播
    3. Fusion: 微观-宏观特征的门控注意力融合

    支持两种微观特征输入模式：
    - 模式 A (向后兼容): 直接传入预融合的 micro_embeddings
    - 模式 B (推荐): 传入 h_visual + h_caption，使用内部 MicroFusionEncoder
    """

    def __init__(
        self,
        micro_feature_dim: int = 128,
        macro_feature_dim: int = 128,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        macro_graph_data=None,
        image_to_street_mapping: torch.Tensor = None,
        # 模式 A: 预融合的微观嵌入
        micro_embeddings: torch.Tensor = None,
        # 模式 B: 分离的视觉和文本特征（内部融合）
        h_visual: torch.Tensor = None,
        h_caption: torch.Tensor = None,
        visual_dim: int = 128,
        caption_dim: int = 384,
        micro_fusion_mode: Literal[
            "gated", "concat", "visual_only", "caption_only"
        ] = "gated",
        gnn_type: str = "GAT",
        gnn_aggr: str = "mean",
    ):
        """
        初始化双层感知模型

        Args:
            micro_feature_dim: 微观特征维度（融合后）
            macro_feature_dim: 宏观特征维度
            hidden_dim: 隐藏层维度
            num_gnn_layers: GAT 层数
            num_heads: 注意力头数
            dropout: Dropout 比例
            macro_graph_data: 宏观图数据（包含 edge_index_topo, edge_index_func）
            image_to_street_mapping: 图像到街道的映射 [num_images]
            micro_embeddings: [模式A] 预融合的微观嵌入 [num_images, micro_feature_dim]
            h_visual: [模式B] 视觉特征 [num_images, visual_dim]
            h_caption: [模式B] 文本特征 [num_images, caption_dim]
            visual_dim: 视觉特征维度（GraphMAE 输出）
            caption_dim: 文本特征维度（Sentence-BERT 输出）
            micro_fusion_mode: 微观融合模式
        """
        super().__init__()

        self.micro_feature_dim = micro_feature_dim
        self.macro_feature_dim = macro_feature_dim
        self.hidden_dim = hidden_dim
        self.micro_fusion_mode = micro_fusion_mode
        self.gnn_type = gnn_type
        self.gnn_aggr = gnn_aggr

        # ====== 微观层：确定特征来源 ======
        self._setup_micro_features(
            micro_embeddings=micro_embeddings,
            h_visual=h_visual,
            h_caption=h_caption,
            visual_dim=visual_dim,
            caption_dim=caption_dim,
            output_dim=micro_feature_dim,
            fusion_mode=micro_fusion_mode,
            dropout=dropout,
        )

        # ====== 宏观层：街道映射和图结构 ======
        self._setup_macro_graph(
            macro_graph_data=macro_graph_data,
            image_to_street_mapping=image_to_street_mapping,
        )

        # ====== 宏观 GNN ======
        if gnn_type != "GAT":
            self.macro_gnn = ConfigurableMacroGNN(
                input_dim=micro_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_gnn_layers,
                num_heads=num_heads,
                dropout=dropout,
                gnn_type=gnn_type,
                gnn_aggr=gnn_aggr,
            )
        else:
            self.macro_gnn = MacroGNN(
                input_dim=micro_feature_dim,
                hidden_dim=hidden_dim,
                num_layers=num_gnn_layers,
                num_heads=num_heads,
                dropout=dropout,
            )

        gnn_out_dim = self.macro_gnn.output_dim

        # ====== 微观-宏观融合 ======
        self.micro_macro_fusion = MicroMacroFusion(
            micro_dim=micro_feature_dim,
            macro_dim=gnn_out_dim,
            hidden_dim=64,
            dropout=dropout,
        )

        self.final_dim = self.micro_macro_fusion.output_dim

        # ====== Bradley-Terry 评分头 ======
        self.score_head = nn.Sequential(
            nn.Linear(self.final_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _setup_micro_features(
        self,
        micro_embeddings: Optional[torch.Tensor],
        h_visual: Optional[torch.Tensor],
        h_caption: Optional[torch.Tensor],
        visual_dim: int,
        caption_dim: int,
        output_dim: int,
        fusion_mode: str,
        dropout: float,
    ):
        """设置微观特征（模式 A 或模式 B）"""

        # 模式 B: 分离特征 + 内部融合
        if h_visual is not None and h_caption is not None:
            self.use_internal_fusion = True
            self.register_buffer("h_visual", h_visual)
            self.register_buffer("h_caption", h_caption)

            # 初始化微观融合编码器
            self.micro_encoder = MicroFusionEncoder(
                visual_dim=visual_dim,
                caption_dim=caption_dim,
                output_dim=output_dim,
                fusion_mode=fusion_mode,
                dropout=dropout,
            )

            # 预计算融合特征并缓存（可选，加速推理）
            with torch.no_grad():
                fused = self.micro_encoder(h_visual, h_caption)
                self.register_buffer("micro_embeddings", fused)

        # 模式 A: 预融合特征
        elif micro_embeddings is not None:
            self.use_internal_fusion = False
            self.micro_encoder = None
            self.register_buffer("micro_embeddings", micro_embeddings)
            self.register_buffer("h_visual", None)
            self.register_buffer("h_caption", None)

        else:
            # 延迟初始化
            self.use_internal_fusion = False
            self.micro_encoder = None
            self.micro_embeddings = None
            self.h_visual = None
            self.h_caption = None

    def _setup_macro_graph(
        self, macro_graph_data, image_to_street_mapping: Optional[torch.Tensor]
    ):
        """设置宏观图结构"""

        if image_to_street_mapping is not None:
            self.register_buffer("image_to_street_mapping", image_to_street_mapping)
        else:
            self.image_to_street_mapping = None

        if macro_graph_data is not None:
            self.register_buffer("topo_edge_index", macro_graph_data.edge_index_topo)

            if (
                hasattr(macro_graph_data, "edge_index_func")
                and macro_graph_data.edge_index_func is not None
            ):
                self.register_buffer(
                    "func_edge_index", macro_graph_data.edge_index_func
                )
            else:
                self.func_edge_index = None

            self.num_streets = macro_graph_data.num_nodes
        else:
            self.topo_edge_index = None
            self.func_edge_index = None
            self.num_streets = 0

    def _get_micro_embeddings(self) -> torch.Tensor:
        """获取微观嵌入（支持动态融合或使用缓存）"""
        if (
            self.use_internal_fusion
            and self.training
            and self.micro_encoder is not None
        ):
            # 训练时动态计算（允许梯度流动）
            return self.micro_encoder(self.h_visual, self.h_caption)
        else:
            # 推理时使用缓存
            return self.micro_embeddings

    def _get_edge_index(self) -> torch.Tensor:
        """获取合并后的边索引（拓扑 + 功能）"""
        edge_index = self.topo_edge_index

        if self.func_edge_index is not None and self.func_edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, self.func_edge_index], dim=1)

        return edge_index

    def _compute_macro_features(self, micro_emb: torch.Tensor) -> torch.Tensor:
        """计算宏观特征：聚合 + GNN 传播"""

        # 聚合微观特征到街道级
        street_features = scatter_mean(
            micro_emb, self.image_to_street_mapping, dim=0, dim_size=self.num_streets
        )
        street_features = torch.nan_to_num(street_features)

        # GNN 传播
        edge_index = self._get_edge_index()
        x_macro = self.macro_gnn(street_features, edge_index)

        return x_macro

    def forward(
        self, left_indices: torch.Tensor, right_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播：Pairwise 比较预测

        Args:
            left_indices: 左图像索引 [batch_size]
            right_indices: 右图像索引 [batch_size]

        Returns:
            prob: P(left > right) 的概率 [batch_size]
        """
        # 1. 获取微观嵌入
        micro_emb = self._get_micro_embeddings()

        # 2. 计算宏观特征
        x_macro = self._compute_macro_features(micro_emb)

        # 3. 获取样本特征
        left_micro = micro_emb[left_indices]
        right_micro = micro_emb[right_indices]

        left_street_idx = self.image_to_street_mapping[left_indices]
        right_street_idx = self.image_to_street_mapping[right_indices]

        left_macro = x_macro[left_street_idx]
        right_macro = x_macro[right_street_idx]

        # 4. 微观-宏观融合
        left_fused = self.micro_macro_fusion(left_micro, left_macro)
        right_fused = self.micro_macro_fusion(right_micro, right_macro)

        # 5. Bradley-Terry 评分
        left_score = self.score_head(left_fused)
        right_score = self.score_head(right_fused)

        # 6. 比较概率
        logits = left_score - right_score
        return torch.sigmoid(logits).squeeze()

    def predict_score(self, indices: torch.Tensor) -> torch.Tensor:
        """
        预测绝对感知分数

        Args:
            indices: 图像索引 [batch_size]

        Returns:
            scores: 感知分数 [batch_size]
        """
        micro_emb = self._get_micro_embeddings()
        x_macro = self._compute_macro_features(micro_emb)

        micro = micro_emb[indices]
        street_idx = self.image_to_street_mapping[indices]
        macro = x_macro[street_idx]

        fused = self.micro_macro_fusion(micro, macro)
        return self.score_head(fused).squeeze()

    def extract_features(self, indices: torch.Tensor) -> torch.Tensor:
        """
        提取融合特征（用于下游任务，如多维度回归）

        Args:
            indices: 图像索引 [batch_size]

        Returns:
            features: 融合特征 [batch_size, final_dim]
        """
        micro_emb = self._get_micro_embeddings()
        x_macro = self._compute_macro_features(micro_emb)

        micro = micro_emb[indices]
        street_idx = self.image_to_street_mapping[indices]
        macro = x_macro[street_idx]

        return self.micro_macro_fusion(micro, macro)

    def extract_features_direct(
        self, micro_emb: torch.Tensor, street_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        对新样本直接提取特征（推理用）

        Args:
            micro_emb: 新样本的微观嵌入 [batch_size, micro_feature_dim]
            street_indices: 新样本对应的街道索引 [batch_size]

        Returns:
            features: 融合特征 [batch_size, final_dim]
        """
        # 使用训练集的宏观图状态
        cached_micro = self._get_micro_embeddings()
        x_macro = self._compute_macro_features(cached_micro)

        # 对新样本融合
        micro = micro_emb.to(x_macro.device)
        street_idx = street_indices.to(x_macro.device)
        macro = x_macro[street_idx]

        return self.micro_macro_fusion(micro, macro)


# ==============================================================================
# 工厂函数：便捷创建模型
# ==============================================================================
def create_dual_layer_model(
    # 微观特征（二选一）
    micro_embeddings: torch.Tensor = None,
    h_visual: torch.Tensor = None,
    h_caption: torch.Tensor = None,
    # 宏观图
    macro_graph_data=None,
    image_to_street_mapping: torch.Tensor = None,
    # 模型配置
    micro_feature_dim: int = 128,
    visual_dim: int = 128,
    caption_dim: int = 384,
    micro_fusion_mode: str = "gated",
    hidden_dim: int = 128,
    num_gnn_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.2,
    gnn_type: str = "GAT",
    gnn_aggr: str = "mean",
) -> DualLayerPerceptionModel:
    """
    便捷工厂函数：创建双层感知模型

    使用示例：

    ```python
    # 模式 A: 使用预融合特征
    model = create_dual_layer_model(
        micro_embeddings=torch.load('fused_representations.pt'),
        macro_graph_data=macro_graph,
        image_to_street_mapping=mapping
    )

    # 模式 B: 使用分离特征（内部融合）
    model = create_dual_layer_model(
        h_visual=torch.load('graph_representations.pt'),
        h_caption=caption_embeddings,
        micro_fusion_mode='gated',
        macro_graph_data=macro_graph,
        image_to_street_mapping=mapping
    )
    ```
    """
    return DualLayerPerceptionModel(
        micro_feature_dim=micro_feature_dim,
        macro_feature_dim=micro_feature_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_gnn_layers,
        num_heads=num_heads,
        dropout=dropout,
        macro_graph_data=macro_graph_data,
        image_to_street_mapping=image_to_street_mapping,
        micro_embeddings=micro_embeddings,
        h_visual=h_visual,
        h_caption=h_caption,
        visual_dim=visual_dim,
        caption_dim=caption_dim,
        micro_fusion_mode=micro_fusion_mode,
        gnn_type=gnn_type,
        gnn_aggr=gnn_aggr,
    )
