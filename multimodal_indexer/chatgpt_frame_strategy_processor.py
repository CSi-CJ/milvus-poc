#!/usr/bin/env python3
"""
ChatGPT推荐的视频帧提取策略处理器
采用 1 FPS 抽帧 + 相似度过滤 + 重点帧识别的完整方案
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from PIL import Image, ImageEnhance
import io
import hashlib
from skimage.metrics import structural_similarity as ssim

class ChatGPTFrameStrategyProcessor:
    """基于ChatGPT推荐的完整帧提取策略处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reader = self._initialize_easyocr()
        
    def _initialize_easyocr(self):
        """初始化EasyOCR"""
        try:
            import easyocr
            reader = easyocr.Reader(['ch_sim', 'en'], verbose=False)
            self.logger.info("✅ EasyOCR初始化成功")
            return reader
        except ImportError:
            self.logger.error("❌ EasyOCR不可用，请安装: pip install easyocr")
            return None
    
    def extract_frames_with_chatgpt_strategy(self, video_path: str) -> List[Dict[str, Any]]:
        """使用ChatGPT推荐的完整策略提取视频帧"""
        if not self.reader:
            return []
        
        self.logger.info("🚀 开始ChatGPT完整帧提取策略...")
        
        # 步骤1: 1 FPS 抽帧策略
        raw_frames = self._extract_1fps_frames(video_path)
        if not raw_frames:
            return []
        
        self.logger.info(f"📊 1 FPS策略提取了 {len(raw_frames)} 帧")
        
        # 步骤2: 相似度过滤
        filtered_frames = self._filter_similar_frames(raw_frames)
        self.logger.info(f"🔍 相似度过滤后剩余 {len(filtered_frames)} 帧 (减少了 {len(raw_frames) - len(filtered_frames)} 帧)")
        
        # 步骤3: 重点帧识别和排序
        prioritized_frames = self._prioritize_frames(filtered_frames)
        self.logger.info(f"⭐ 重点帧识别完成，按重要性排序")
        
        # 步骤4: OCR文本提取
        ocr_results = self._extract_text_from_prioritized_frames(prioritized_frames)
        
        return ocr_results
    
    def _extract_1fps_frames(self, video_path: str) -> List[Dict[str, Any]]:
        """步骤1: 采用 1 FPS 抽帧策略"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"❌ 无法打开视频文件: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            self.logger.info(f"📹 视频信息: {duration:.1f}秒, {fps:.1f} FPS, {frame_count} 总帧数")
            
            frames = []
            frame_interval = int(fps)  # 1 FPS = 每秒1帧
            
            for frame_idx in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 转换为高质量PNG
                    success, buffer = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    if success:
                        timestamp = frame_idx / fps if fps > 0 else 0
                        frames.append({
                            'frame_number': len(frames),
                            'original_frame_idx': frame_idx,
                            'timestamp': timestamp,
                            'frame_data': buffer.tobytes(),
                            'frame_array': frame.copy()  # 保存用于相似度计算
                        })
                        self.logger.debug(f"提取帧 {frame_idx} (时间: {timestamp:.1f}s)")
            
            cap.release()
            self.logger.info(f"✅ 1 FPS策略完成，提取 {len(frames)} 帧")
            return frames
            
        except Exception as e:
            self.logger.error(f"❌ 1 FPS帧提取失败: {e}")
            return []
    
    def _filter_similar_frames(self, frames: List[Dict[str, Any]], 
                             ssim_threshold: float = 0.95, 
                             phash_threshold: int = 5) -> List[Dict[str, Any]]:
        """步骤2: 过滤重复或相似帧"""
        if len(frames) <= 1:
            return frames
        
        filtered_frames = [frames[0]]  # 总是保留第一帧
        
        for i, current_frame in enumerate(frames[1:], 1):
            is_similar = False
            
            # 与已选择的帧进行相似度比较
            for selected_frame in filtered_frames:
                # 方法1: SSIM结构相似度
                similarity = self._calculate_ssim(
                    selected_frame['frame_array'], 
                    current_frame['frame_array']
                )
                
                if similarity > ssim_threshold:
                    is_similar = True
                    self.logger.debug(f"帧 {i} 与已选帧相似 (SSIM: {similarity:.3f})")
                    break
                
                # 方法2: 感知哈希
                hash_distance = self._calculate_phash_distance(
                    selected_frame['frame_array'], 
                    current_frame['frame_array']
                )
                
                if hash_distance < phash_threshold:
                    is_similar = True
                    self.logger.debug(f"帧 {i} 与已选帧相似 (Hash距离: {hash_distance})")
                    break
            
            if not is_similar:
                filtered_frames.append(current_frame)
                self.logger.debug(f"保留帧 {i} (时间: {current_frame['timestamp']:.1f}s)")
        
        # 清理frame_array以节省内存
        for frame in filtered_frames:
            if 'frame_array' in frame:
                del frame['frame_array']
        
        return filtered_frames
    
    def _calculate_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """计算两帧之间的结构相似度"""
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 调整尺寸以提高计算速度
            height, width = gray1.shape
            if width > 640:
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray1 = cv2.resize(gray1, (new_width, new_height))
                gray2 = cv2.resize(gray2, (new_width, new_height))
            
            # 计算SSIM
            similarity = ssim(gray1, gray2)
            return similarity
            
        except Exception as e:
            self.logger.warning(f"SSIM计算失败: {e}")
            return 0.0
    
    def _calculate_phash_distance(self, frame1: np.ndarray, frame2: np.ndarray) -> int:
        """计算两帧之间的感知哈希距离"""
        try:
            def perceptual_hash(frame):
                # 转换为灰度图并调整大小
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (8, 8))
                
                # 计算平均值
                avg = resized.mean()
                
                # 生成哈希
                hash_bits = []
                for row in resized:
                    for pixel in row:
                        hash_bits.append('1' if pixel > avg else '0')
                
                return ''.join(hash_bits)
            
            hash1 = perceptual_hash(frame1)
            hash2 = perceptual_hash(frame2)
            
            # 计算汉明距离
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            return distance
            
        except Exception as e:
            self.logger.warning(f"感知哈希计算失败: {e}")
            return 100  # 返回大值表示不相似
    
    def _prioritize_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤3: 重点帧识别和优先级排序"""
        for frame in frames:
            score = 0
            
            # 加载帧数据进行分析
            frame_array = self._load_frame_from_bytes(frame['frame_data'])
            if frame_array is None:
                frame['priority_score'] = 0
                continue
            
            # 评分标准1: 文字密度估算
            text_density = self._estimate_text_density(frame_array)
            score += text_density * 3  # 文字密度权重最高
            
            # 评分标准2: UI元素数量
            ui_elements = self._count_ui_elements(frame_array)
            score += ui_elements * 2
            
            # 评分标准3: 对比度和清晰度
            clarity = self._assess_frame_clarity(frame_array)
            score += clarity * 1.5
            
            # 评分标准4: 边缘密度（UI界面通常边缘丰富）
            edge_density = self._calculate_edge_density(frame_array)
            score += edge_density * 1
            
            frame['priority_score'] = score
            frame['text_density'] = text_density
            frame['ui_elements'] = ui_elements
            frame['clarity'] = clarity
            frame['edge_density'] = edge_density
            
            self.logger.debug(f"帧 {frame['frame_number']} 评分: {score:.2f} "
                            f"(文字:{text_density:.2f}, UI:{ui_elements:.2f}, "
                            f"清晰度:{clarity:.2f}, 边缘:{edge_density:.2f})")
        
        # 按优先级排序
        prioritized = sorted(frames, key=lambda x: x['priority_score'], reverse=True)
        
        self.logger.info("🏆 帧优先级排序完成:")
        for i, frame in enumerate(prioritized[:5]):  # 显示前5个
            self.logger.info(f"  {i+1}. 帧{frame['frame_number']} (时间:{frame['timestamp']:.1f}s) "
                           f"评分:{frame['priority_score']:.2f}")
        
        return prioritized
    
    def _load_frame_from_bytes(self, frame_data: bytes) -> np.ndarray:
        """从字节数据加载帧"""
        try:
            image = Image.open(io.BytesIO(frame_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.warning(f"帧数据加载失败: {e}")
            return None
    
    def _estimate_text_density(self, frame: np.ndarray) -> float:
        """估算帧中的文字密度"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用形态学操作检测文字区域
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # 边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 形态学闭运算连接文字
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # 计算文字区域比例
            text_pixels = np.sum(closed > 0)
            total_pixels = closed.shape[0] * closed.shape[1]
            
            density = text_pixels / total_pixels
            return min(density * 10, 10)  # 归一化到0-10
            
        except Exception as e:
            self.logger.warning(f"文字密度估算失败: {e}")
            return 0
    
    def _count_ui_elements(self, frame: np.ndarray) -> float:
        """计算UI元素数量"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 检测矩形区域（按钮、卡片等）
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ui_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 50000:  # 过滤太小或太大的区域
                    # 检查是否接近矩形
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    if len(approx) >= 4:  # 类似矩形的形状
                        ui_count += 1
            
            return min(ui_count / 5, 10)  # 归一化到0-10
            
        except Exception as e:
            self.logger.warning(f"UI元素计数失败: {e}")
            return 0
    
    def _assess_frame_clarity(self, frame: np.ndarray) -> float:
        """评估帧的清晰度"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 使用拉普拉斯算子计算清晰度
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 归一化到0-10
            clarity = min(laplacian_var / 100, 10)
            return clarity
            
        except Exception as e:
            self.logger.warning(f"清晰度评估失败: {e}")
            return 0
    
    def _calculate_edge_density(self, frame: np.ndarray) -> float:
        """计算边缘密度"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            
            density = edge_pixels / total_pixels
            return min(density * 20, 10)  # 归一化到0-10
            
        except Exception as e:
            self.logger.warning(f"边缘密度计算失败: {e}")
            return 0
    
    def _extract_text_from_prioritized_frames(self, frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """步骤4: 从优先级排序的帧中提取OCR文本"""
        results = []
        
        for frame in frames:
            self.logger.info(f"🔄 处理优先级帧 {frame['frame_number']} "
                           f"(评分: {frame['priority_score']:.2f}, 时间: {frame['timestamp']:.1f}s)")
            
            try:
                # 增强帧质量
                enhanced_frame_data = self._enhance_frame_for_ocr(frame['frame_data'])
                
                # 转换为numpy数组
                image = Image.open(io.BytesIO(enhanced_frame_data))
                image_array = np.array(image)
                
                # 使用EasyOCR识别
                ocr_results = self.reader.readtext(image_array)
                
                # 提取文字和置信度
                texts = []
                confidences = []
                
                for (bbox, text, prob) in ocr_results:
                    if prob > 0.5:  # 置信度阈值
                        texts.append(text)
                        confidences.append(prob)
                        self.logger.debug(f"识别文字: {text} (置信度: {prob:.3f})")
                
                # 合并文本
                combined_text = '\n'.join(texts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                result = {
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'text': combined_text,
                    'confidence': avg_confidence,
                    'priority_score': frame['priority_score'],
                    'text_density': frame.get('text_density', 0),
                    'ui_elements': frame.get('ui_elements', 0),
                    'clarity': frame.get('clarity', 0),
                    'engine': 'easyocr_chatgpt_strategy',
                    'method': 'chatgpt_frame_strategy'
                }
                
                results.append(result)
                
                self.logger.info(f"✅ 帧 {frame['frame_number']} 完成，"
                               f"提取文本: {len(combined_text)} 字符，置信度: {avg_confidence:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ 帧 {frame['frame_number']} 处理失败: {e}")
                results.append({
                    'frame_number': frame['frame_number'],
                    'timestamp': frame['timestamp'],
                    'text': '',
                    'confidence': 0.0,
                    'priority_score': frame.get('priority_score', 0),
                    'engine': 'easyocr_chatgpt_strategy',
                    'error': str(e)
                })
        
        return results
    
    def _enhance_frame_for_ocr(self, frame_data: bytes) -> bytes:
        """增强帧质量以提升OCR效果"""
        try:
            # 转换为OpenCV格式
            image = Image.open(io.BytesIO(frame_data))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 1. 尺寸优化
            height, width = cv_image.shape[:2]
            if width < 1920:
                scale_factor = min(2.5, 1920 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                cv_image = cv2.resize(cv_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # 2. 去噪
            denoised = cv2.bilateralFilter(cv_image, 9, 75, 75)
            
            # 3. 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # 4. 对比度增强
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 5. PIL优化
            pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
            
            # 对比度增强
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.3)
            
            # 锐度增强
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.4)
            
            # 转换回字节数据
            enhanced_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.png', enhanced_cv, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            if success:
                return buffer.tobytes()
            else:
                return frame_data
                
        except Exception as e:
            self.logger.warning(f"帧增强失败: {e}")
            return frame_data


def test_chatgpt_frame_strategy():
    """测试ChatGPT帧提取策略"""
    import os
    
    print("🧪 测试ChatGPT完整帧提取策略")
    print("="*60)
    
    processor = ChatGPTFrameStrategyProcessor()
    
    if not processor.reader:
        print("❌ EasyOCR不可用，请先安装: pip install easyocr")
        return
    
    # 测试视频文件
    test_video = "./files/个性化推荐.mp4"
    
    if not os.path.exists(test_video):
        print(f"❌ 测试视频文件不存在: {test_video}")
        return
    
    try:
        print(f"🔄 开始处理视频: {test_video}")
        results = processor.extract_frames_with_chatgpt_strategy(test_video)
        
        print(f"\n📋 ChatGPT完整策略OCR结果:")
        print("="*60)
        
        total_text_length = 0
        successful_frames = 0
        
        for result in results:
            frame_num = result['frame_number']
            text = result['text']
            confidence = result['confidence']
            priority = result.get('priority_score', 0)
            timestamp = result.get('timestamp', 0)
            
            print(f"\n🖼️  帧 {frame_num} (时间: {timestamp:.1f}s, 优先级: {priority:.2f}):")
            print(f"   策略: ChatGPT完整帧提取策略")
            print(f"   置信度: {confidence:.3f}")
            print(f"   文本长度: {len(text)} 字符")
            
            if text:
                print(f"   📝 提取文本:")
                print("   " + "-" * 50)
                for line in text.split('\n'):
                    if line.strip():
                        print(f"   {line}")
                print("   " + "-" * 50)
                total_text_length += len(text)
                successful_frames += 1
            else:
                print("   ⚠️  未提取到文本")
        
        print(f"\n📊 总结:")
        print(f"   处理帧数: {len(results)}")
        print(f"   成功提取文本的帧数: {successful_frames}")
        print(f"   总文本长度: {total_text_length} 字符")
        print(f"   平均每帧文本长度: {total_text_length/successful_frames if successful_frames > 0 else 0:.1f} 字符")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_chatgpt_frame_strategy()