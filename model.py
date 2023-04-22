import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d


def knn(x, centroids, k):
    """
    Input:
        x: pointcloud data, [B, N, F]
        centroids: sampled pointcloud data, [B, npoint, F]
        k: number of nearest neighbors to return
    Return:
        knn_indices: indices of the k nearest neighbors for each npoint in centroids, [B, npoint, k]
    """
    distances = torch.cdist(x, centroids) # 返回欧几里得距离的平方
    # knn_indices, knn_distances = torch.topk(distances, k=k, dim=1, largest=False)
    knn_indices = torch.topk(distances, k=k, dim=1,  
                             largest=False).indices  # [B, k, npoint]
    return knn_indices.permute(0, 2, 1).contiguous()  # [B, npoint, k]


def farthest_point_sample(x, xyz, npoint):
    """
    Input:
        x: pointcloud data, [B, N, F]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint, F]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    F = x.shape[-1]
    # 将centroids从[B, npoint]扩展成[B, npoint, F]
    centroids_expand = centroids.unsqueeze(-1).repeat(1, 1, F)
    centroids_expand_xyz = centroids.unsqueeze(-1).repeat(1, 1, 3)
    # 按维度1从x中收集元素，得到[B, npoint, F]的结果
    centroids = torch.gather(x, 1, centroids_expand)
    xyz = torch.gather(xyz, 1, centroids_expand_xyz)

    return centroids, xyz

def get_graph_feature(x, k=20, idx=None):
    batch_size, num_points, num_dims = x.shape
    if idx is None:
        idx = knn(x, x, k=k)   # (batch_size, num_points, k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    x = x.contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=-1).permute(0, 3, 1, 2).contiguous()
  
    return feature

def get_fpfh_feature(x, xyz, k=20, idx=None, npoint=1024):
    # x是[B, N, F],B为batch_size,N为点数，F为特征维度
    # centroids是[B, npoint, F]，npoint为采样点数
    # idx得到的是[B, npoint, k],采样点的k个邻居的索引
    # N1个点的特征为[X,Xk-X]两者拼接而成，即总特征变为[B,N1,(Xk-X,X),k]
    # 最大池化，[B,N1,(Xk-X,X),k]->[B,N1,(Xk-X,X)]

    batch_size = x.size(0)
    num_points = x.size(1)
    if idx is None:
        centroids, new_xyz = farthest_point_sample(x, xyz, npoint)
        idx = knn(xyz, new_xyz, k=k)

    device = x.device

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, _, num_dims = centroids.size()

    x = x.contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, npoint, k, num_dims)
    centroids = centroids.view(
        batch_size, npoint, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((centroids, feature-centroids),
                        dim=-1).permute(0, 3, 1, 2).contiguous()

    return feature, new_xyz

def cat_feature(x, xyz_high, xyz_low):
    # x是[B, N, F],B为batch_size,N为点数，F为特征维度
    # xyz_high高分辨率点云的坐标，xyz_low低分辨率点云的坐标
    # 得到低分辨率点云对应的高分辨率点云的特征

    batch_size = x.size(0)
    num_points = x.size(1)
    npoint = xyz_low.size(1)

    idx = knn(xyz_high, xyz_low, k=1)

    device = x.device

    idx_base = torch.arange(
        0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    x = x.contiguous() 
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, npoint, -1)

    return feature


### ######################### ###
# o3d版本:不可GPU加速，循环计算
### ######################### ###

# 计算FPFH特征
def Compute_FPFH(xyz, k=20):
    """
    Input:
    xyz: pointcloud data, [B, N, 3], B为batch_size,N为点数,3为特征维度(x,y,z)
    k: number of nearest neighbors
    Return:
    FPFH: The feature of each point, [B, N, 33]
    """
    B, N, _ = xyz.shape
    FPFH = torch.zeros(B, N, 33)

    for b in range(B):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz[b].cpu().numpy())

        # 计算法线
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(k))

        # 计算每个点的FPFH特征
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud, o3d.geometry.KDTreeSearchParamKNN(k))

        # 将FPFH特征转换为tensor
        fpfh_tensor = torch.tensor(
            fpfh.data, dtype=torch.float32).permute(1, 0)

        FPFH[b] = fpfh_tensor

    return FPFH

# -----------------------------------------

class Rand_FPFH(nn.Module):
    def __init__(self, args, output_channels=40):
        super(Rand_FPFH, self).__init__()
        self.args = args
        # DGCNN层
        self.graph_conv = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv1d(36, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 第二层
        self.conv2 = nn.Sequential(
            nn.Conv1d(97, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 第三层
        self.conv3 = nn.Sequential(
            nn.Conv1d(161, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 第四层
        self.conv4 = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        device = x.device

        # Graph Feature
        x = x.permute(0, 2, 1).contiguous()  # (B, N, 3)
        xyz = x[:, :, :3].contiguous()
        graph_feature = get_graph_feature(x)  # (B, 6, N, k)
        graph_xyz = xyz
        graph_feature = self.graph_conv(graph_feature) # (B, 64, N, k)
        graph_feature = graph_feature.max(dim=-1, keepdim=False)[0] # (B, 64, N)

        # FPFH Feature
        fpfh = Compute_FPFH(xyz, k=20).to(device)  # (B, N, 33)
        x1 = torch.cat((x, fpfh), dim=-1).permute(0, 2, 1).contiguous()  # (B, 36, N)
        x1 = self.conv1(x1) # (B, 32, N)
        x1, xyz1 = get_fpfh_feature(x1.permute(0,2,1), xyz, k=20, idx=None, npoint=1024) # (B, 64, N1, k)
        x1 = x1.max(dim=-1, keepdim=False)[0] # (B, 64, N1)

        x1 = x1.permute(0, 2, 1).contiguous()  # (B, N1, 64)
        fpfh = Compute_FPFH(xyz1, k=20).to(device)  # (B, N1, 33)
        x2 = torch.cat((x1, fpfh), dim=-1).permute(0, 2, 1).contiguous()  # (B, 97, N1)
        x2 = self.conv2(x2) # (B, 64, N1)
        x2, xyz2 = get_fpfh_feature(x2.permute(0,2,1), xyz1, k=20, idx=None, npoint=512) # (B, 128, N2, k)
        x2 = x2.max(dim=-1, keepdim=False)[0] # (B, 128, N2)

        x2 = x2.permute(0, 2, 1).contiguous()  # (B, N2, 128)
        fpfh = Compute_FPFH(xyz2, k=20).to(device)  # (B, N2, 33)
        x3 = torch.cat((x2, fpfh), dim=-1).permute(0, 2, 1).contiguous() # (B, 161, N2)
        x3 = self.conv3(x3) # (B, 128, N2)
        x3, xyz3 = get_fpfh_feature(x3.permute(0,2,1), xyz2, k=20, idx=None, npoint=256) # (B, 256, N3, k)
        x3 = x3.max(dim=-1, keepdim=False)[0] # (B, 256, N3)

        graph_feature_=cat_feature(graph_feature.permute(0,2,1),graph_xyz,xyz3) # (B, N3, 64)
        x1_=cat_feature(x1,xyz1,xyz3) # (B, N3, 64)
        x2_=cat_feature(x2,xyz2,xyz3) # (B, N3, 128)

        feature = torch.cat((graph_feature_,x1_,x2_,x3),dim=-1).permute(0,2,1).contiguous() # (B, 512, N3)

        batch_size = feature.size(0)
        feature = self.conv4(feature)
        feature_max = F.adaptive_max_pool1d(feature, 1).view(batch_size, -1)
        feature_avg = F.adaptive_avg_pool1d(feature, 1).view(batch_size, -1)
        feature = torch.cat((feature_max, feature_avg), 1)

        feature = F.leaky_relu(self.bn1(self.linear1(feature)), negative_slope=0.2)
        feature = self.dp1(feature)
        feature = F.leaky_relu(self.bn2(self.linear2(feature)), negative_slope=0.2)
        feature = self.dp2(feature)
        feature = self.linear3(feature)

        return feature
