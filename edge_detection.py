import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import filters
import os

class CellEdgeDetector:
    """
    细胞图像边缘检测器类
    实现多种边缘检测算法用于HEp-2细胞图像分析
    """
    
    def __init__(self, image_path):
        """
        初始化边缘检测器
        
        参数:
            image_path: 图像文件路径
        """
        # 读取彩色图像和灰度图像
        self.color_image = cv2.imread(image_path) # 读取彩色图像
        self.image = cv2.imread(image_path,0) # 读取灰度图像,把彩色图像转换为灰度图像

        # 检查图像是否成功读取
        if self.image is None or self.color_image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 将BGR转换为RGB格式用于显示
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)

    
    def denoise_image(self, image, method='gaussian', kernel_size=5, strength=1):
        """
        图像降噪处理
        
        参数:
            image: ���图像
            method: 降噪方法 ('gaussian', 'median', 'bilateral')
            kernel_size: 核大小
            strength: 降噪强度
        """
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), strength)
        elif method == 'median':
            return cv2.medianBlur(image, kernel_size)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, kernel_size, strength * 75, strength * 75)
        return image


    def robinson_detector(self, denoise=False, denoise_params=None, threshold=0, normalize=True):
        """
        Robinson边缘检测
        
        参数:
            denoise: 是否进行降噪
            denoise_params: 降噪参数字典
            threshold: 边缘阈值
            normalize: 是否归一化结果
        """
        # 处理图像副本
        processed_image = self.image.copy()
        
        # 应用降噪
        if denoise:
            denoise_params = denoise_params or {}
            processed_image = self.denoise_image(processed_image, **denoise_params)
            
        # Robinson算子
        kernel_x = np.array([[-1, 1, 1], 
                           [-1, -2, 1], 
                           [-1, 1, 1]])
        
        kernel_y = np.array([[1, 1, 1], 
                           [1, -2, 1], 
                           [-1, -1, -1]])
        
        grad_x = signal.convolve2d(processed_image, kernel_x, mode='same')
        grad_y = signal.convolve2d(processed_image, kernel_y, mode='same')
        
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用阈值
        if threshold > 0:
            edges[edges < threshold] = 0
            
        return self.normalize_image(edges) if normalize else edges
    
    def sobel_detector(self, denoise=False, denoise_params=None, threshold=0):
        """
        Sobel边缘检测
        """
        # 处理图像副本
        processed_image = self.image.copy()
        
        # 应用降噪
        if denoise:
            denoise_params = denoise_params or {}
            processed_image = self.denoise_image(processed_image, **denoise_params)
            
        # Sobel算子
        kernel_x = np.array([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], 
                           [0, 0, 0], 
                           [-1, -2, -1]])
        
        grad_x = signal.convolve2d(processed_image, kernel_x, mode='same')
        grad_y = signal.convolve2d(processed_image, kernel_y, mode='same')
        
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用阈值
        if threshold > 0:
            edges[edges < threshold] = 0
            
        return self.normalize_image(edges)
    
    def prewitt_detector(self, denoise=False, denoise_params=None, threshold=0):
        """
        Prewitt边缘检测
        
        参数:
            denoise: 是否进行降噪
            denoise_params: 降噪参数字典
            threshold: 边缘阈值
        """
        # 处理图像副本
        processed_image = self.image.copy()
        
        # 应用降噪
        if denoise:
            denoise_params = denoise_params or {}
            processed_image = self.denoise_image(processed_image, **denoise_params)
            
        # Prewitt算子
        kernel_x = np.array([[-1, 0, 1], 
                           [-1, 0, 1], 
                           [-1, 0, 1]])
        kernel_y = np.array([[1, 1, 1], 
                           [0, 0, 0], 
                           [-1, -1, -1]])
        
        grad_x = signal.convolve2d(processed_image, kernel_x, mode='same')
        grad_y = signal.convolve2d(processed_image, kernel_y, mode='same')
        
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用阈值
        if threshold > 0:
            edges[edges < threshold] = 0
            
        return self.normalize_image(edges)
    
    def kirsch_detector(self, denoise=False, denoise_params=None, threshold=0):
        """
        Kirsch边缘检测
        
        参数:
            denoise: 是否进行降噪
            denoise_params: 降噪参数字典
            threshold: 边缘阈值
        """
        # 处理图像副本
        processed_image = self.image.copy()
        
        # 应用降噪
        if denoise:
            denoise_params = denoise_params or {}
            processed_image = self.denoise_image(processed_image, **denoise_params)
            
        # Kirsch算子
        kernel_x = np.array([[-5, 3, 3], 
                           [-5, 0, 3], 
                           [-5, 3, 3]])
        kernel_y = np.array([[3, 3, 3], 
                           [3, 0, 3], 
                           [-5, -5, -5]])
        
        grad_x = signal.convolve2d(processed_image, kernel_x, mode='same')
        grad_y = signal.convolve2d(processed_image, kernel_y, mode='same')
        
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用阈值
        if threshold > 0:
            edges[edges < threshold] = 0
            
        return self.normalize_image(edges)
    
    def gaussian_detector(self, denoise=False, denoise_params=None, threshold=0):
        """
        高斯边缘检测
        
        参数:
            denoise: 是否进行降噪
            denoise_params: 降噪参数字典
            threshold: 边缘阈值
        """
        # 处理图像副本
        processed_image = self.image.copy()
        
        # 应用降噪
        if denoise:
            denoise_params = denoise_params or {}
            processed_image = self.denoise_image(processed_image, **denoise_params)
            
        # 高斯算子
        gaussian_kernel_x = np.array([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ]) / 240

        gaussian_kernel_y = np.array([
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ]) / 240

        grad_x = signal.convolve2d(processed_image, gaussian_kernel_x, mode='same')
        grad_y = signal.convolve2d(processed_image, gaussian_kernel_y, mode='same')
        
        edges = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用阈值
        if threshold > 0:
            edges[edges < threshold] = 0
            
        return self.normalize_image(edges)
    
    def canny_detector(self, low_threshold=65, high_threshold=100):
        """
        Canny边缘检测
        """
        edges = cv2.Canny(self.image, low_threshold, high_threshold)
        return edges
    
    @staticmethod 
    def normalize_image(image):
        """
        归一化图像到0-255范围
        """
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def show_results(self, save_path=None, robinson_params=None, sobel_params=None, 
                    prewitt_params=None, kirsch_params=None, gaussian_params=None, 
                    canny_params=None):
        """
        显示所有边缘检测结果的对比
        """
        # 生成所有检测结果
        results = {
            'Color': self.color_image,
            'Grayscale': self.image,
            'Robinson': self.robinson_detector(
                denoise=robinson_params.get('denoise', False),
                denoise_params=robinson_params.get('denoise_params'),
                threshold=robinson_params.get('threshold', 0)
            ) if robinson_params else self.robinson_detector(),
            'Sobel': self.sobel_detector(
                denoise=sobel_params.get('denoise', False),
                denoise_params=sobel_params.get('denoise_params'),
                threshold=sobel_params.get('threshold', 0)
            ) if sobel_params else self.sobel_detector(),
            'Prewitt': self.prewitt_detector(
                denoise=prewitt_params.get('denoise', False),
                denoise_params=prewitt_params.get('denoise_params'),
                threshold=prewitt_params.get('threshold', 0)
            ) if prewitt_params else self.prewitt_detector(),
            'Kirsch': self.kirsch_detector(
                denoise=kirsch_params.get('denoise', False),
                denoise_params=kirsch_params.get('denoise_params'),
                threshold=kirsch_params.get('threshold', 0)
            ) if kirsch_params else self.kirsch_detector(),
            'Gaussian': self.gaussian_detector(
                denoise=gaussian_params.get('denoise', False),
                denoise_params=gaussian_params.get('denoise_params'),
                threshold=gaussian_params.get('threshold', 0)
            ) if gaussian_params else self.gaussian_detector(),
            'Canny': self.canny_detector(
                **(canny_params or {'low_threshold': 65, 'high_threshold': 100})
            )
        }
        
        # 创建子图布局并设置更大的间距
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # 增加子图间距
        fig.suptitle('Edge Detection Results Comparison', fontsize=16, y=1.05)  # 调整标题位置
        
        # 展平axes数组以便迭代
        axes_flat = axes.flatten()
        
        # 显示每个结果
        for idx, (title, image) in enumerate(results.items()):
            ax = axes_flat[idx]
            try:
                if title == 'Color':
                    ax.imshow(image)
                else:
                    # 确保图像数据类型正确
                    if isinstance(image, np.ndarray):
                        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
                    else:
                        print(f"警告: {title} 图像格式不正确")
                        continue
                ax.set_title(title, pad=10)  # 增加标题和图像的间距
                ax.axis('off')
            except Exception as e:
                print(f"显示 {title} 图像时出错: {e}")

        # 调整布局
        plt.tight_layout()
        
        # 保存或显示结果
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                print(f"图像已保存至: {save_path}")
            except Exception as e:
                print(f"保存图像时出错: {e}")
            finally:
                plt.close()
        else:
            plt.show()

def main():
    """
    主函数
    """
    try:
        # 创建results文件夹（如果不存在）
        if not os.path.exists('results'):
            os.makedirs('results')
            
        # 处理三张图像
        image_paths = [
            'Data 06 30213/cells/9343 AM.bmp',
            'Data 06 30213/cells/10905 JL.bmp',
            'Data 06 30213/cells/43590 AM.bmp'
        ]
        
        # 为每种图像定制参数
        params_list = [
            {  # 第一张图片的参数
                'robinson_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'gaussian', 'kernel_size': 5, 'strength': 1}
                },
                'sobel_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'bilateral', 'kernel_size': 3, 'strength': 1.5}
                },
                'prewitt_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'median', 'kernel_size': 3, 'strength': 1}
                },
                'kirsch_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'gaussian', 'kernel_size': 5, 'strength': 1.2}
                },
                'gaussian_params': {
                    'threshold': 0,
                    'denoise': False,
                    'denoise_params': {'method': 'bilateral', 'kernel_size': 5, 'strength': 1.8}
                },
                'canny_params': {'low_threshold': 40, 'high_threshold': 180}
            },
            {  # 第二张图片的参数
                'robinson_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'gaussian', 'kernel_size': 5, 'strength': 1}
                },
                'sobel_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'bilateral', 'kernel_size': 3, 'strength': 1.5}
                },
                'prewitt_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'median', 'kernel_size': 3, 'strength': 1}
                },
                'kirsch_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'gaussian', 'kernel_size': 5, 'strength': 1.2}
                },
                'gaussian_params': {
                    'threshold': 0,
                    'denoise': False,
                    'denoise_params': {'method': 'bilateral', 'kernel_size': 5, 'strength': 1.8}
                },
                'canny_params': {'low_threshold': 30, 'high_threshold': 120}
            },
            {  # 第三张图片的参数
                'robinson_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'gaussian', 'kernel_size': 5, 'strength': 1}
                },
                'sobel_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'bilateral', 'kernel_size': 3, 'strength': 1.5}
                },
                'prewitt_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'median', 'kernel_size': 3, 'strength': 1}
                },
                'kirsch_params': {
                    'threshold': 10,
                    'denoise': True,
                    'denoise_params': {'method': 'gaussian', 'kernel_size': 5, 'strength': 1.2}
                },
                'gaussian_params': {
                    'threshold': 0,
                    'denoise': False,
                    'denoise_params': {'method': 'bilateral', 'kernel_size': 5, 'strength': 1.8}
                },
                'canny_params': {'low_threshold': 10, 'high_threshold': 60}
            }
        ]
        
        for image_path, params in zip(image_paths, params_list):
            print(f"\n处理图像: {image_path}")
            detector = CellEdgeDetector(image_path)
            
            filename = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join('results', f'{filename}_results.png')
            
            detector.show_results(save_path=save_path, **params)
            print(f"结果已保存至: {save_path}")
            
    except Exception as e:
        print(f"处理过程中出错: {e}")






if __name__ == "__main__":
    main() 


