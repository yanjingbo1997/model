import os
import warnings
import random
import concurrent.futures
from functools import lru_cache
import gc
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import time as time_module
import tempfile



class ImprovedGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.3, n_layers=3, residual=True, n_heads=4):
        super(ImprovedGCN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.residual = residual
        self.n_heads = n_heads
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        for _ in range(n_layers - 2):  # 中间层
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers - 1)])
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim // 4),
                nn.LeakyReLU(0.2),
                nn.Linear(output_dim // 4, 1)
            ) for _ in range(n_heads)
        ])
        self.attn_combine = nn.Linear(n_heads, 1, bias=False)
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        # 特征投影
        h = self.feature_proj(x)
        h = F.leaky_relu(h, 0.2)
        h = F.dropout(h, p=self.dropout, training=self.training)
        prev_h = h
        for i in range(self.n_layers - 1):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.leaky_relu(h, 0.2)
            if self.residual and h.size() == prev_h.size():
                h = h + prev_h
            h = F.dropout(h, p=self.dropout, training=self.training)
            prev_h = h

        h = self.convs[-1](h, edge_index)
        attn_weights = []
        for head in self.attention:
            weights = head(h)
            attn_weights.append(F.softmax(weights, dim=0))
        multi_head_weights = torch.cat(attn_weights, dim=1)
        combined_weights = torch.sigmoid(self.attn_combine(multi_head_weights))
        h = h * combined_weights

        return h


class MemoryEfficientDetector:

    def __init__(self, hidden_dim=16, output_dim=8, learning_rate=0.001, num_epochs=30):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.embeddings = {}
        self.communities = {}

        print(f"Using device: {self.device}")


    def _prepare_graph_data_enhanced(self, graph, use_cached=True, cache_file=None):

        node_list = list(graph.nodes())
        n_nodes = len(node_list)

        if cache_file is None:
            graph_name = getattr(graph, 'name', 'unknown')
            graph_size_hash = hash(f"{n_nodes}_{graph.number_of_edges()}")
            cache_file = f"./feature_cache/{graph_name}_{n_nodes}nodes_{abs(graph_size_hash)}.npz"

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if use_cached and os.path.exists(cache_file):
            try:
                cached_data = np.load(cache_file, mmap_mode='r')
                node_features = cached_data['features']

                if node_features.shape[0] == n_nodes:
                    x = torch.FloatTensor(node_features.copy())
                else:
                    x = self._compute_node_features_batched(graph, node_list)
                    np.savez(cache_file, features=x.numpy())
            except Exception as e:
                x = self._compute_node_features_batched(graph, node_list)
                np.savez(cache_file, features=x.numpy())
        else:
            x = self._compute_node_features_batched(graph, node_list)
            if cache_file:
                np.savez(cache_file, features=x.numpy())

        node_to_idx = {node: i for i, node in enumerate(node_list)}
        edge_index = []

        batch_size = 100000
        edge_iterator = iter(graph.edges())
        current_batch = []

        while True:
            try:
                for _ in range(batch_size):
                    u, v = next(edge_iterator)
                    current_batch.append((node_to_idx[u], node_to_idx[v]))
            except StopIteration:
                break
            finally:
                if current_batch:
                    for u_idx, v_idx in current_batch:
                        edge_index.append([u_idx, v_idx])
                    current_batch = []
                    gc.collect()

        if edge_index:
            edge_index = torch.LongTensor(edge_index).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index)

        gc.collect()

        return data, node_list

    def _compute_node_features_batched(self, graph, node_list):

        n_nodes = len(node_list)

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        features = np.memmap(temp_file.name, dtype=np.float32, mode='w+', shape=(n_nodes, n_features))
        batch_size = 10000

        for start_idx in range(0, n_nodes, batch_size):
            end_idx = min(start_idx + batch_size, n_nodes)
            current_nodes = node_list[start_idx:end_idx]
            for i, node in enumerate(current_nodes):
                idx = start_idx + i
                features[idx, 0] = graph.in_degree(node)
                features[idx, 1] = graph.out_degree(node)
                features[idx, 2] = features[idx, 0] + features[idx, 1]
            features.flush()
            gc.collect()

            if (end_idx - start_idx) > 1000:
                print(f"处理节点度: {end_idx}/{n_nodes}")
            result = np.array(features)
            del features
            os.unlink(temp_file.name)
            return torch.FloatTensor(result)

        for start_idx in range(0, n_nodes, batch_size):
            end_idx = min(start_idx + batch_size, n_nodes)
            current_nodes = node_list[start_idx:end_idx]
            sub_clustering = nx.clustering(undirected_graph, nodes=current_nodes)
            for i, node in enumerate(current_nodes):
                idx = start_idx + i
                features[idx, 3] = sub_clustering.get(node, 0)
            features.flush()
            gc.collect()

            if (end_idx - start_idx) > 1000:
                print(f"处理聚类系数: {end_idx}/{n_nodes}")

        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)


        for i, node in enumerate(node_list):
            features[i, 4] = pagerank.get(node, 0)
        features.flush()
        for j in range(n_features):
            col_min = np.min(features[:, j])
            col_max = np.max(features[:, j])
            if col_max - col_min > 0:
                features[:, j] = (features[:, j] - col_min) / (col_max - col_min)
        result = np.array(features)

        del features
        os.unlink(temp_file.name)

        print(f"计算了{n_nodes}个节点的{n_features}个特征")
        return torch.FloatTensor(result)

    def _compute_node_features(self, graph, node_list):

        n_nodes = len(node_list)
        print(f"计算{n_nodes}个节点的特征（处理所有节点）...")

        n_features = 5

        feature_cache_dir = "./feature_cache"
        os.makedirs(feature_cache_dir, exist_ok=True)

        graph_name = getattr(graph, 'name', 'graph')
        graph_size_hash = hash(f"{n_nodes}_{graph.number_of_edges()}")
        cache_file = f"{feature_cache_dir}/{graph_name}_{n_nodes}nodes_{abs(graph_size_hash)}.npz"

        if os.path.exists(cache_file):
            print(f"加载已缓存的节点特征: {cache_file}")
            cached_data = np.load(cache_file, mmap_mode='r')
            features_shape = cached_data['features'].shape

            if features_shape[0] == n_nodes and features_shape[1] == n_features:
                features = np.array(cached_data['features'])
                print(f"成功加载特征: 形状 {features_shape}")
                return torch.FloatTensor(features)
            else:
                print(f"缓存特征不匹配: 形状 {features_shape} vs 期望的 ({n_nodes}, {n_features})")

        print(f"计算{n_nodes}个节点的{n_features}个特征...")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        features = np.memmap(temp_file.name, dtype=np.float32, mode='w+', shape=(n_nodes, n_features))

        batch_size = 10000

        print("计算度特征...")

        undirected_graph = graph.to_undirected()

        for start_idx in range(0, n_nodes, batch_size):
            end_idx = min(start_idx + batch_size, n_nodes)
            current_nodes = node_list[start_idx:end_idx]

            for i, node in enumerate(current_nodes):
                idx = start_idx + i
                # 入度
                features[idx, 0] = graph.in_degree(node)
                # 出度
                features[idx, 1] = graph.out_degree(node)
                # 总度
                features[idx, 2] = features[idx, 0] + features[idx, 1]

            features.flush()

            gc.collect()

            if (end_idx - start_idx) > 1000:
                print(f"处理节点度: {end_idx}/{n_nodes}")

        print("计算聚类系数...")

        for start_idx in range(0, n_nodes, batch_size):
            end_idx = min(start_idx + batch_size, n_nodes)
            current_nodes = node_list[start_idx:end_idx]

            try:
                sub_clustering = nx.clustering(undirected_graph, nodes=current_nodes)

                for i, node in enumerate(current_nodes):
                    idx = start_idx + i
                    features[idx, 3] = sub_clustering.get(node, 0)
            except Exception as e:
                print(f"计算聚类系数时出错: {e}")
                for i in range(start_idx, end_idx):
                    features[i, 3] = 0

            features.flush()

            gc.collect()

            if (end_idx - start_idx) > 1000:
                print(f"处理聚类系数: {end_idx}/{n_nodes}")

        print("计算PageRank值...")

        try:

            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)

            # 填充PageRank值
            for i, node in enumerate(node_list):
                features[i, 4] = pagerank.get(node, 0)
        except Exception as e:
            print(f"计算PageRank时出错: {e}")

        features.flush()

        print("规范化特征...")
        for j in range(n_features):
            col_min = np.min(features[:, j])
            col_max = np.max(features[:, j])
            if col_max - col_min > 0:
                features[:, j] = (features[:, j] - col_min) / (col_max - col_min)

        print(f"将特征保存到缓存: {cache_file}")
        np.savez(cache_file, features=features)

        result = np.array(features)

        del features
        os.unlink(temp_file.name)

        print(f"成功计算并保存了{n_nodes}个节点的{n_features}个特征")
        return torch.FloatTensor(result)


    def _prepare_graph_data(self, graph):

        node_list = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        degrees = np.array([graph.degree(node) for node in node_list])
        x = torch.FloatTensor(degrees.reshape(-1, 1))

        edge_index = []
        for u, v in graph.edges():
            edge_index.append([node_to_idx[u], node_to_idx[v]])

        if edge_index:
            edge_index = torch.LongTensor(np.array(edge_index).T)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index)

        return data, node_list

    def negative_sampling_efficient(self, pos_edge_index, num_nodes, num_samples=1000):
        neg_edges = set()
        max_attempts = num_samples * 2
        attempts = 0

        pos_edges_set = set()
        for i in range(pos_edge_index.size(1)):
            u, v = pos_edge_index[0, i].item(), pos_edge_index[1, i].item()
            pos_edges_set.add((u, v))

        while len(neg_edges) < num_samples and attempts < max_attempts:
            batch_size = 1000
            u = np.random.randint(0, num_nodes, batch_size)
            v = np.random.randint(0, num_nodes, batch_size)

            for i in range(batch_size):
                if u[i] != v[i] and (u[i], v[i]) not in pos_edges_set and (u[i], v[i]) not in neg_edges:
                    neg_edges.add((u[i], v[i]))
                    if len(neg_edges) >= num_samples:
                        break

            attempts += batch_size

        if neg_edges:
            neg_edges_list = list(neg_edges)
            u = [e[0] for e in neg_edges_list]
            v = [e[1] for e in neg_edges_list]
            neg_edge_index = torch.tensor([u, v], dtype=torch.long)
        else:
            neg_edge_index = torch.zeros((2, 0), dtype=torch.long)

        return neg_edge_index.to(self.device)

    def detect_communities(self, layer_name, time_windows, method='dbscan'):

        print(f"检测{layer_name}层的社区...")
        window_communities = {}

        for window_name, graph in time_windows.items():
            cache_file = f"./feature_cache/{layer_name}_{window_name}_features.npz"
            data, node_ids = self._prepare_graph_data_enhanced(graph, use_cached=True, cache_file=cache_file)

            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)

            if edge_index.size(1) == 0:
                embeddings = x.cpu().numpy()
                if method == 'dbscan':
                    labels, n_clusters = self._perform_dbscan_clustering(embeddings)
                else:
                    labels, n_clusters = self._absolute_minimal_clustering(embeddings)

                window_communities[window_name] = {
                    'labels': labels,
                    'node_ids': node_ids,
                    'n_clusters': n_clusters
                }
                continue

            input_dim = x.size(1)
            model = ImprovedGCN(
                input_dim=input_dim,
                hidden_dim=128,
                output_dim=64,
                dropout=0.3,
                n_layers=3,
                residual=True,
                n_heads=4
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

            model.train()

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

            patience = 10
            patience_counter = 0
            best_loss = float('inf')

            batch_size = max(1, min(10000, edge_index.size(1)))

            for epoch in range(self.num_epochs):
                total_loss = 0
                n_batches = 0

                perm = torch.randperm(edge_index.size(1))
                edge_index_shuffled = edge_index[:, perm]

                for i in range(0, edge_index_shuffled.size(1), batch_size):
                    end = min(i + batch_size, edge_index_shuffled.size(1))
                    batch_edges = edge_index_shuffled[:, i:end]

                    optimizer.zero_grad()
                    embeddings = model(x, batch_edges)
                    pos_edge_index = batch_edges
                    neg_edge_index = self.negative_sampling_efficient(pos_edge_index, x.size(0))
                    pos_score = F.cosine_similarity(embeddings[pos_edge_index[0]], embeddings[pos_edge_index[1]], dim=1)
                    neg_score = F.cosine_similarity(embeddings[neg_edge_index[0]], embeddings[neg_edge_index[1]], dim=1)
                    margin = 0.5
                    pos_loss = torch.mean(torch.clamp(margin - pos_score, min=0))
                    neg_loss = torch.mean(torch.clamp(neg_score + margin, min=0))
                    loss = pos_loss + neg_loss
                    if edge_index.size(1) > 0:
                        sampled_nodes = torch.randint(0, x.size(0), (min(1000, x.size(0)),), device=self.device)
                        node_embeddings = embeddings[sampled_nodes]
                        sim_matrix = torch.mm(node_embeddings, node_embeddings.t())
                        struct_loss = torch.mean(
                            torch.abs(sim_matrix - torch.eye(sim_matrix.size(0), device=self.device)))
                        loss += 0.1 * struct_loss

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                avg_loss = total_loss / n_batches if n_batches > 0 else float('inf')
                scheduler.step(avg_loss)

                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1}/{self.num_epochs}, Avg Loss: {avg_loss:.4f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            model.eval()

            embeddings_file = f"./feature_cache/{layer_name}_{window_name}_embeddings.npy"

            if os.path.exists(embeddings_file):
                embeddings = np.load(embeddings_file)
            else:
                embeddings = np.zeros((x.size(0), model.output_dim), dtype=np.float32)
                batch_size = 5000

                with torch.no_grad():
                    for i in range(0, x.size(0), batch_size):
                        end = min(i + batch_size, x.size(0))
                        batch_x = x[i:end]
                        batch_edge_index = None
                        try:
                            mask = ((edge_index[0] >= i) & (edge_index[0] < end)) | (
                                    (edge_index[1] >= i) & (edge_index[1] < end))
                            batch_edge_index = edge_index[:, mask]
                            batch_edge_index = batch_edge_index.clone()
                            mask_0 = batch_edge_index[0] >= i
                            mask_1 = batch_edge_index[1] >= i
                            batch_edge_index[0, mask_0] = batch_edge_index[0, mask_0] - i
                            batch_edge_index[1, mask_1] = batch_edge_index[1, mask_1] - i
                            mask = (batch_edge_index[0] < end - i) & (batch_edge_index[1] < end - i)
                            batch_edge_index = batch_edge_index[:, mask]
                        except Exception as e:
                            batch_embeddings = model.feature_proj(batch_x)
                            batch_embeddings = F.leaky_relu(batch_embeddings, 0.2)
                            embeddings[i:end] = batch_embeddings.cpu().numpy()
                            continue
                        if batch_edge_index is not None and batch_edge_index.size(1) > 0:
                            batch_embeddings = model(batch_x, batch_edge_index)
                        else:
                            batch_embeddings = model.feature_proj(batch_x)
                            batch_embeddings = F.leaky_relu(batch_embeddings, 0.2)

                        embeddings[i:end] = batch_embeddings.cpu().numpy()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                np.save(embeddings_file, embeddings)

            print("执行社区检测...")
            if method == 'dbscan':
                labels, n_clusters = self._perform_dbscan_clustering(embeddings)
            else:
                print(f"未知方法 '{method}'，使用DBSCAN")
                labels, n_clusters = self._perform_dbscan_clustering(embeddings)

            window_communities[window_name] = {
                'labels': labels,
                'node_ids': node_ids,
                'n_clusters': n_clusters
            }

            print(f"窗口 {window_name}: 检测到 {n_clusters} 个社区")

            del data, x, edge_index, model
            if 'embeddings' in locals():
                del embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # 保存社区结果
        self.communities[layer_name] = window_communities
        return window_communities