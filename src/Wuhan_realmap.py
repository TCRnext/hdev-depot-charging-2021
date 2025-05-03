import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from pylab import mpl

class WuhanMapProcessor:
    def __init__(self, map_path, center_pixel=(1024, 1024), center_coord=(114.3055, 30.5928), px_per_km=60):
        """
        初始化地图处理器
        
        参数:
        - map_path: 地图图片路径
        - center_pixel: 地图中心点像素坐标 (x, y)
        - center_coord: 地图中心点经纬度 (lon, lat)
        - px_per_km: 每公里对应的像素数
        """
        self.map_img = Image.open(map_path)
        self.center_pixel = np.array(center_pixel)
        self.center_coord = np.array(center_coord)
        self.px_per_km = px_per_km
        self.km_per_px = 1 / px_per_km
        self.points = []  # 存储带标签的点
        self.add_point(114.2391, 30.6451, "常青花园", 0, 3)
        self.add_point(114.32393, 30.650516, "百步亭", 0, 3)
        self.add_point(114.424164, 30.607484, "武汉站", 1, 3)
        self.add_point(114.307867, 30.608112, "武汉天地", 2, 3)
        self.add_point(114.301524, 30.598348, "三阳路", 2, 2)
        self.add_point(114.294637, 30.577479, "江汉路A", 1, 4)
        self.add_point(114.287298, 30.576448, "江汉路B", 1, 4)
        self.add_point(114.288838, 30.58284, "江汉路C", 1, 4)
        self.add_point(114.27538, 30.582784, "中山大道A", 1, 4)
        self.add_point(114.270822, 30.577682, "中山大道B", 1, 4)
        self.add_point(114.276221, 30.567782, "汉正街", 1, 4)
        self.add_point(114.255889, 30.616392, "汉口站", 1, 3)
        self.add_point(114.26508, 30.631113, "石桥", 0, 3)
        self.add_point(114.245839, 30.599543, "武汉CBD", 2, 2)
        self.add_point(114.305063, 30.537111, "首义广场", 2, 4)
        self.add_point(114.300349, 30.545157, "黄鹤楼", 1, 2)
        self.add_point(114.318425, 30.545044, "小东门", 2, 2)
        self.add_point(114.308676, 30.561356, "积玉桥", 2, 2)
        self.add_point(114.342389, 30.555528, "楚河汉街", 1, 4)
        self.add_point(114.317603, 30.528545, "武昌站", 1, 3)
        self.add_point(114.331925, 30.536768, "中南路", 1, 3)
        self.add_point(114.35232, 30.520379, "武汉理工大学", 0, 4)
        self.add_point(114.398466, 30.505589, "光谷广场", 2, 4)
        self.add_point(114.390237, 30.499225, "雄楚大道", 0, 4)
        self.add_point(114.421755, 30.495645, "关山大道", 0, 4)
        self.add_point(114.385415, 30.475751, "中南财大", 0, 3)
        self.add_point(114.44977, 30.504377, "光谷东", 0, 2)
        self.add_point(114.289952, 30.477258, "白沙洲", 2, 4)
        self.add_point(114.284648, 30.49166, "张家湾", 0, 2)
        self.add_point(114.214714, 30.597339, "古田四路", 2, 3)
        self.add_point(114.208793, 30.561982, "王家湾", 2, 3)
        self.add_point(114.26578, 30.549481, "钟家村", 2, 3)
        self.add_point(114.24394, 30.518988, "国博", 1, 2)
        self.add_point(114.203158, 30.497534, "江城大道", 0, 3)
        self.add_point(114.165837, 30.500508, "东风公司", 1, 3)
        self.add_point(114.167961, 30.619003, "竹叶海", 1, 3)
        self.add_point(114.219232, 30.660015, "金银湖", 0, 3)
        self.add_point(114.348037, 30.586934, "徐东大街", 2, 4)
        self.add_point(114.368738, 30.583544, "铁机路", 2, 3)
        # 假设经度1度≈111km*cos(lat)，纬度1度≈111km
        self.lon_scale = 111 * np.cos(np.radians(center_coord[1]))  # km/度
        self.lat_scale = 111  # km/度
    
    def add_point(self, lon, lat, label,point_type,weight=1):
        """
        添加带标签的点
        type: 0 rx, 1 tx, 2 mixed
        """
        self.points.append({'coord': (lon, lat), 'label': label, 'type': point_type, 'weight': weight})
    
    def coord_to_xy(self, lon, lat):
        """
        将经纬度坐标转换为平面直角坐标 (km)
        
        返回:
        - (x, y): 东方向为x+，北方向为y+
        """
        # 计算与中心点的经纬度差
        delta_lon = lon - self.center_coord[0]
        delta_lat = lat - self.center_coord[1]
        
        # 转换为km
        x = delta_lon * self.lon_scale
        y = delta_lat * self.lat_scale
        
        return x, y
    
    def xy_to_pixel(self, x, y):
        """
        将平面直角坐标(km)转换为像素坐标
        
        返回:
        - (px, py): 图片像素坐标
        """
        # 转换为相对于中心点的像素偏移
        dx = x * self.px_per_km
        dy = -y * self.px_per_km  # 图片y轴向下为正
        
        # 转换为绝对像素坐标
        px = self.center_pixel[0] + dx
        py = self.center_pixel[1] + dy
        
        return int(round(px)), int(round(py))
    
    def coord_to_pixel(self, lon, lat):
        """直接将经纬度转换为像素坐标"""
        x, y = self.coord_to_xy(lon, lat)
        return self.xy_to_pixel(x, y)
    
    def get_submap(self, center_lon, center_lat, radius_km):
        """
        获取子地图
        
        参数:
        - center_lon: 子地图中心经度
        - center_lat: 子地图中心纬度
        - radius_km: 半径(km)，子地图边长=radius_km*2
        
        返回:
        - submap_img: 子地图图片
        - points_in_range: 范围内的点(已转换为子地图坐标系)
        """
        # 计算子地图中心在原始地图中的像素位置
        center_x, center_y = self.coord_to_xy(center_lon, center_lat)
        center_px, center_py = self.xy_to_pixel(center_x, center_y)
        
        # 计算子地图半径(像素)
        radius_px = int(round(radius_km * self.px_per_km))
        
        # 计算子地图边界
        left = center_px - radius_px
        top = center_py - radius_px
        right = center_px + radius_px
        bottom = center_py + radius_px
        
        # 确保边界在图像范围内
        width, height = self.map_img.size
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        
        # 截取子地图
        submap_img = self.map_img.crop((left, top, right, bottom))
        
        # 找出范围内的点
        points_in_range = []
        for point in self.points:
            px, py = self.coord_to_pixel(point['coord'][0], point['coord'][1])
            
            # 检查点是否在子地图范围内
            if left <= px < right and top <= py < bottom:
                # 转换为子地图坐标系 (km)
                submap_x = (px - left) / self.px_per_km - radius_km
                submap_y = (radius_km - (py - top) / self.px_per_km)
                if submap_x**2 + submap_y**2 <= radius_km**2:
                    # 添加到结果列表
                    points_in_range.append({
                        'original_coord': point['coord'],
                        'label': point['label'],
                        'x': submap_x,
                        'y': submap_y,
                        'type': point['type'],
                        'weight': point['weight']
                    })
        
        return submap_img, points_in_range
    
    def create_heatmap(self, submap_img, labeled_points, heatmap_points, radius_km):
        """
        创建带热力图和标签的地图
        
        参数:
        - submap_img: 子地图图片
        - labeled_points: 带标签的点列表 [{'x': x, 'y': y, 'label': text}, ...]
        - heatmap_points: 仅用于热力图的点列表 [(x1, y1), (x2, y2), ...]
        - radius_km: 子地图半径(km)
        
        返回:
        - 合成后的图片
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 显示底图
        ax.imshow(submap_img, extent=[-radius_km, radius_km, -radius_km, radius_km])
        
        # 绘制热力图

        if heatmap_points and len(heatmap_points) > 0:
                
            heatmap_x = [p[0] for p in heatmap_points]
            heatmap_y = [p[1] for p in heatmap_points]
            
            # 创建热力图数据
            heatmap, xedges, yedges = np.histogram2d(
                heatmap_x, heatmap_y, 
                bins=100, 
                range=[[-radius_km, radius_km], [-radius_km, radius_km]]
            )
            
            
            # 平滑处理
            heatmap = gaussian_filter(heatmap, sigma=1)

            
            # 自定义颜色映射 (从透明到红色)
            cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, 1, 1, 0), (1, 0, 0, 1)])
            
            # 显示热力图
            ax.imshow(
                heatmap.T, extent=[-radius_km, radius_km, -radius_km, radius_km],
                origin='lower', cmap=cmap, alpha=0.9
            )

        
        # 绘制带标签的点
        for point in labeled_points:
            if point['type'] == 0:
                color = 'green'
            elif point['type'] == 1:
                color = 'red'
            else:
                color = 'blue'
            
            ax.plot(point['x'], point['y'], 'bo', markersize=5, color=color)
            ax.text(
                point['x'], point['y'] + 0.1, point['label'],
                ha='center', va='bottom', color='blue', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
        
        # 绘制表示半径的圆
        circle = plt.Circle((0, 0), radius_km, color='r', fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circle)
        
        ax.set_xlim(-radius_km, radius_km)
        ax.set_ylim(-radius_km, radius_km)
        ax.set_xlabel('East (km)')
        ax.set_ylabel('North (km)')
        ax.set_title(f'Map with Radius {radius_km} km')
        
        # 转换为PIL图像返回
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(img)


# 使用示例
if __name__ == "__main__":
    # 初始化地图处理器
    mpl.rcParams["font.sans-serif"] = ["SimHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    processor = WuhanMapProcessor("hdev-depot-charging-2021/img_src/wuhan_map.png")
    
    
    # 获取子地图
    submap, points_in_range = processor.get_submap(114.27538, 30.582784, 8)  # 2km半径
    
    print("Points in range:")
    for p in points_in_range:
        print(f"{p['label']}: ({p['x']:.2f}, {p['y']:.2f}) km")
    
    # 创建一些热力图数据点 (随机生成示例)
    heatmap_points = []
    for _ in range(100):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, 1.5)
        heatmap_points.append((x, y))
    
    # 创建合成地图
    result_img = processor.create_heatmap(
        submap, 
        points_in_range, 
        heatmap_points, 
        8
    )
    
    # 保存结果
    result_img.save("result_map.png")
    print("Result map saved to result_map.png")