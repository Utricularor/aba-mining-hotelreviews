"""Contrastive learning module for node representation optimization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ContrastiveLearning(nn.Module):
    """Contrastive learning for node embeddings."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        temperature: float = 0.07,
        use_projection_head: bool = True
    ):
        """
        Initialize contrastive learning module.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temperature: Temperature for contrastive loss
            use_projection_head: Whether to use projection head
        """
        super(ContrastiveLearning, self).__init__()
        
        self.temperature = temperature
        self.use_projection_head = use_projection_head
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Projection head for contrastive learning
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim // 2)
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Encoded representations
        """
        h = self.encoder(x)
        
        if self.use_projection_head and self.training:
            z = self.projection_head(h)
            return h, z
        
        return h
    
    def compute_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z1: First view embeddings
            z2: Second view embeddings
            labels: Optional labels for supervised contrastive
            
        Returns:
            Contrastive loss
        """
        batch_size = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )
        
        # Create positive mask
        if labels is not None:
            # Supervised contrastive
            labels = torch.cat([labels, labels], dim=0)
            mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask = mask.float()
        else:
            # Self-supervised contrastive
            mask = torch.zeros((2 * batch_size, 2 * batch_size), device=z1.device)
            mask[torch.arange(batch_size), batch_size + torch.arange(batch_size)] = 1
            mask[batch_size + torch.arange(batch_size), torch.arange(batch_size)] = 1
        
        # Remove diagonal
        mask = mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix / self.temperature)
        exp_sim = exp_sim.masked_fill(torch.eye(2 * batch_size, device=z1.device).bool(), 0)
        
        positive_sim = (exp_sim * mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)
        
        loss = -torch.log(positive_sim / (all_sim + 1e-8))
        
        return loss.mean()


class GraphContrastiveLearning(nn.Module):
    """Graph contrastive learning for node embeddings."""
    
    def __init__(
        self,
        encoder: nn.Module,
        temperature: float = 0.07,
        augmentation_prob: float = 0.2
    ):
        """
        Initialize graph contrastive learning.
        
        Args:
            encoder: Graph encoder (e.g., GCN, RGCN)
            temperature: Temperature for contrastive loss
            augmentation_prob: Probability for edge/feature augmentation
        """
        super(GraphContrastiveLearning, self).__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        self.augmentation_prob = augmentation_prob
        
    def augment_graph(
        self,
        edge_index: torch.Tensor,
        x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Augment graph structure and features.
        
        Args:
            edge_index: Edge connectivity
            x: Node features
            
        Returns:
            Augmented edge_index and features
        """
        # Edge dropping
        num_edges = edge_index.shape[1]
        edge_mask = torch.rand(num_edges) > self.augmentation_prob
        aug_edge_index = edge_index[:, edge_mask]
        
        # Feature masking
        aug_x = x
        if x is not None:
            feature_mask = torch.rand_like(x) > self.augmentation_prob
            aug_x = x * feature_mask
        
        return aug_edge_index, aug_x
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with augmentation.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_type: Edge types
            
        Returns:
            Two views of node embeddings
        """
        # Generate two augmented views
        aug_edge_index1, aug_x1 = self.augment_graph(edge_index, x)
        aug_edge_index2, aug_x2 = self.augment_graph(edge_index, x)
        
        # Encode both views
        z1 = self.encoder.encode(aug_x1, aug_edge_index1, edge_type)
        z2 = self.encoder.encode(aug_x2, aug_edge_index2, edge_type)
        
        return z1, z2
    
    def compute_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for graph contrastive learning.
        
        Args:
            z1: First view embeddings
            z2: Second view embeddings
            batch_size: Optional batch size for sampling
            
        Returns:
            Contrastive loss
        """
        # Sample nodes if batch_size is specified
        if batch_size is not None and batch_size < z1.shape[0]:
            indices = torch.randperm(z1.shape[0])[:batch_size]
            z1 = z1[indices]
            z2 = z2[indices]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity
        similarity = torch.mm(z1, z2.t()) / self.temperature
        
        # Create labels (diagonal elements are positives)
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=z1.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class SimCLR(nn.Module):
    """SimCLR implementation for node embeddings."""
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.5
    ):
        """
        Initialize SimCLR.
        
        Args:
            encoder: Base encoder
            projection_dim: Projection head dimension
            temperature: Temperature parameter
        """
        super(SimCLR, self).__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Determine encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(2, encoder.num_nodes)
            dummy_edge = torch.tensor([[0], [1]], dtype=torch.long)
            encoder_output = encoder.encode(dummy_input, dummy_edge)
            encoder_dim = encoder_output.shape[-1]
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            edge_type: Edge types
            
        Returns:
            Representations and projections
        """
        h = self.encoder.encode(x, edge_index, edge_type)
        z = self.projection_head(h)
        
        return h, z
    
    def nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalized temperature-scaled cross entropy loss.
        
        Args:
            z1: First view projections
            z2: Second view projections
            
        Returns:
            NT-Xent loss
        """
        batch_size = z1.shape[0]
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Compute similarity matrix
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(representations, representations.t())
        
        # Create mask for positive pairs
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=z1.device)
        mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)
        
        # Compute loss
        positives = similarity_matrix[mask].view(batch_size * 2, -1)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=z1.device)
        
        logits = logits / self.temperature
        loss = F.cross_entropy(logits, labels)
        
        return loss