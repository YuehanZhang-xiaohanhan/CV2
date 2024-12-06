import cv2
import numpy as np
import matplotlib.pyplot as plt

class LineDetector:
    """
    图像直线检测类
    使用Canny边缘检测和霍夫变换来检测图像中的直线
    """
    
    def __init__(self, image_path):
        """
        初始化检测器
        
        参数:
            image_path: 图像文件路径
        """
        # 读取图像
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Unable to read image: {image_path}")
            
        # 将图像转换为RGB格式用于显示
        self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        # 将图像转换为灰度格式
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
    def detect_edges(self, low_threshold=60, high_threshold=160):
        """
        使用Canny算法检测边缘
        
        参数:
            low_threshold: Canny算法的低阈值
            high_threshold: Canny算法的高阈值
        返回:
            边缘检测后的图像
        """
        # 使用高斯滤波减少噪声
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        # 应用Canny边缘检测
        self.edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return self.edges
    
    def detect_lines(self, rho=1, theta=np.pi/90, threshold=120,
                    min_line_length=80, max_line_gap=8):
        """
        使用霍夫变换检测直线
        
        参数:
            rho: 距离分辨率（像素）
            theta: 角度分辨率（弧度）
            threshold: 累加器阈值
            min_line_length: 最小线段长度
            max_line_gap: 最大线段间隙
        返回:
            带有检测线条的图像
        """
        # 检查是否已经进行过边缘检测
        if not hasattr(self, 'edges'):
            self.detect_edges()
            
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(self.edges, rho, theta, threshold,
                               minLineLength=min_line_length,
                               maxLineGap=max_line_gap)
        
        # 创建结果图像的副本
        result_image = self.display_image.copy()
        
        # 在图像上绘制检测到的线条
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
        return result_image, len(lines) if lines is not None else 0
    
    def show_results(self, save_path=None):
        """
        显示原始图像、边缘检测和直线检测的结果
        
        参数:
            save_path: 保存结果的路径（可选）
        """
        # 获取边缘检测结果
        edge_image = self.detect_edges()
        # 获取直线检测结果
        line_image, num_lines = self.detect_lines()
        
        # 创建图像显示
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示原始图像
        axes[0].imshow(self.display_image)
        axes[0].set_title('original image')
        axes[0].axis('off')
        
        # 显示边缘检测结果
        axes[1].imshow(edge_image, cmap='gray')
        axes[1].set_title('Canny')
        axes[1].axis('off')
        
        # 显示直线检测结果
        axes[2].imshow(line_image)
        axes[2].set_title(f'Hough lines detection\n(detected {num_lines} lines)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path)
            print(f"结果已保存至: {save_path}")
            
        plt.show()
    
    def try_different_params(self, save_path=None):
        """
        尝试不同的参数组合来检测直线
        
        参数:
            save_path: 保存结果的路径（可选）
        """
        # 参数组合
        params = [
            # [rho, theta, threshold, min_line_length, max_line_gap]
            [1, np.pi/180, 100, 100, 8],    # 默认参数
            [1, np.pi/180, 100, 90, 8],     # 更精细的检测
            [1, np.pi/180, 100, 80, 9],     # 只检测明显的长线条
            [2, np.pi/90, 40, 80, 15]       # 更宽松的检测
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for idx, param in enumerate(params):
            rho, theta, threshold, min_line_length, max_line_gap = param
            result_image, num_lines = self.detect_lines(
                rho=rho, 
                theta=theta, 
                threshold=threshold,
                min_line_length=min_line_length, 
                max_line_gap=max_line_gap
            )
            
            axes[idx].imshow(result_image)
            # 简化标签显示
            axes[idx].set_title(f'Detected {num_lines} lines')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path)
            print(f"参数对比结果已保存至: {save_path}")
            
        plt.show()

def main():
    """
    主函数：用于演示直线检测功能
    """
    try:
        # 创建results文件夹（如果不存在）
        import os
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 初始化检测器
        detector = LineDetector('Data 06 30213\\Bhamimage.jpeg')
        # 添加保存路径
        detector.try_different_params(save_path='results/parameter_comparison.png')
        
    except Exception as e:
        print(f"处理过程中出错: {e}")

if __name__ == "__main__":
    main() 