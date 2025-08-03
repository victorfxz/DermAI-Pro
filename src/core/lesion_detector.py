#!/usr/bin/env python3
"""
DermAI Pro - Lesion Detection System
Professional skin lesion detection and segmentation

🔬 Advanced computer vision for lesion identification
🎯 Medical-grade image processing algorithms
🏥 Professional diagnostic support
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
from scipy import ndimage
from skimage import filters, segmentation, measure, morphology
from skimage.feature import local_binary_pattern

logger = logging.getLogger("DermAI-Pro.Lesion-Detector")

class ProfessionalLesionDetector:
    """
    Professional-grade lesion detection system for dermatological analysis
    
    Uses advanced computer vision algorithms to identify and segment
    skin lesions with medical precision.
    """
    
    def __init__(self):
        """Initialize the professional lesion detector"""
        self.initialized = True
        self.detection_methods = [
            'color_segmentation',
            'texture_analysis', 
            'edge_detection',
            'morphological_analysis'
        ]
        
        logger.info("🔬 Professional Lesion Detector initialized")
    
    def detect_lesions(self, image: np.ndarray, detection_mode: str = "comprehensive") -> Dict[str, Any]:
        """
        Detect skin lesions in dermatological image
        
        Args:
            image: Input dermatological image
            detection_mode: "comprehensive", "fast", or "precise"
            
        Returns:
            Dictionary with detection results and lesion information
        """
        try:
            start_time = time.time()
            logger.info(f"🔍 Starting {detection_mode} lesion detection...")
            
            # Validate input image
            if image is None or image.size == 0:
                return self._create_error_result("Invalid input image")
            
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Detect lesions using multiple methods
            lesions_found = []
            
            if detection_mode in ["comprehensive", "precise"]:
                # Use all detection methods
                color_lesions = self._detect_by_color_segmentation(processed_image)
                texture_lesions = self._detect_by_texture_analysis(processed_image)
                edge_lesions = self._detect_by_edge_detection(processed_image)
                morph_lesions = self._detect_by_morphological_analysis(processed_image)
                
                # Combine and validate lesions
                all_lesions = color_lesions + texture_lesions + edge_lesions + morph_lesions
                lesions_found = self._validate_and_merge_lesions(all_lesions, processed_image)
                
            else:  # fast mode
                # Use primary detection method only
                lesions_found = self._detect_by_color_segmentation(processed_image)
            
            # Analyze detected lesions
            analyzed_lesions = []
            for i, lesion in enumerate(lesions_found):
                lesion_analysis = self._analyze_lesion_properties(lesion, processed_image, i+1)
                analyzed_lesions.append(lesion_analysis)
            
            processing_time = time.time() - start_time
            
            logger.info(f"✅ Detection completed: {len(analyzed_lesions)} lesions found in {processing_time:.2f}s")
            
            return {
                'success': True,
                'lesions_detected': len(analyzed_lesions),
                'lesions': analyzed_lesions,
                'detection_mode': detection_mode,
                'processing_time': processing_time,
                'image_dimensions': image.shape,
                'methods_used': self.detection_methods if detection_mode != "fast" else ['color_segmentation']
            }
            
        except Exception as e:
            logger.error(f"❌ Lesion detection failed: {e}")
            return self._create_error_result(str(e))
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal lesion detection"""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                processed = image.copy()
            
            # Noise reduction
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
            
            # Enhance contrast
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
            processed = cv2.merge([l, a, b])
            processed = cv2.cvtColor(processed, cv2.COLOR_LAB2RGB)
            
            return processed
            
        except Exception as e:
            logger.warning(f"⚠️ Image preprocessing failed: {e}")
            return image
    
    def _detect_by_color_segmentation(self, image: np.ndarray) -> List[Dict]:
        """Detect lesions using color-based segmentation"""
        try:
            lesions = []
            
            # Convert to different color spaces for analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Define color ranges for different lesion types
            color_ranges = [
                # Dark lesions (melanoma, moles)
                {'name': 'dark', 'hsv_lower': (0, 50, 0), 'hsv_upper': (180, 255, 100)},
                # Red lesions (inflammation, vascular)
                {'name': 'red', 'hsv_lower': (0, 100, 100), 'hsv_upper': (10, 255, 255)},
                # Brown lesions (pigmented)
                {'name': 'brown', 'hsv_lower': (10, 50, 50), 'hsv_upper': (25, 255, 200)}
            ]
            
            for color_range in color_ranges:
                # Create mask for color range
                mask = cv2.inRange(hsv, color_range['hsv_lower'], color_range['hsv_upper'])
                
                # Morphological operations to clean mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 100 < area < 50000:  # Filter by size
                        # Get bounding box
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        lesions.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'detection_method': 'color_segmentation',
                            'color_type': color_range['name'],
                            'mask': mask[y:y+h, x:x+w]
                        })
            
            return lesions
            
        except Exception as e:
            logger.warning(f"⚠️ Color segmentation failed: {e}")
            return []
    
    def _detect_by_texture_analysis(self, image: np.ndarray) -> List[Dict]:
        """Detect lesions using texture analysis"""
        try:
            lesions = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Local Binary Pattern for texture analysis
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate texture variance
            texture_var = filters.rank.variance(gray, morphology.disk(5))
            
            # Threshold for abnormal texture
            texture_threshold = np.percentile(texture_var, 85)
            texture_mask = texture_var > texture_threshold
            
            # Find connected components
            labeled_mask = measure.label(texture_mask)
            regions = measure.regionprops(labeled_mask)
            
            for region in regions:
                if 100 < region.area < 20000:  # Filter by size
                    minr, minc, maxr, maxc = region.bbox
                    
                    lesions.append({
                        'bbox': (minc, minr, maxc-minc, maxr-minr),
                        'area': region.area,
                        'detection_method': 'texture_analysis',
                        'texture_score': np.mean(texture_var[minr:maxr, minc:maxc]),
                        'region': region
                    })
            
            return lesions
            
        except Exception as e:
            logger.warning(f"⚠️ Texture analysis failed: {e}")
            return []
    
    def _detect_by_edge_detection(self, image: np.ndarray) -> List[Dict]:
        """Detect lesions using edge detection"""
        try:
            lesions = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(gray, 50, 150)
            edges2 = cv2.Canny(gray, 100, 200)
            
            # Combine edges
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Morphological closing to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 30000:  # Filter by size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate edge density
                    roi_edges = edges[y:y+h, x:x+w]
                    edge_density = np.sum(roi_edges > 0) / (w * h)
                    
                    if edge_density > 0.1:  # Minimum edge density
                        lesions.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'detection_method': 'edge_detection',
                            'edge_density': edge_density
                        })
            
            return lesions
            
        except Exception as e:
            logger.warning(f"⚠️ Edge detection failed: {e}")
            return []
    
    def _detect_by_morphological_analysis(self, image: np.ndarray) -> List[Dict]:
        """Detect lesions using morphological analysis"""
        try:
            lesions = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Top-hat transform to detect dark lesions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            
            # Threshold
            _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 150 < area < 25000:  # Filter by size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    lesions.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'detection_method': 'morphological_analysis'
                    })
            
            return lesions
            
        except Exception as e:
            logger.warning(f"⚠️ Morphological analysis failed: {e}")
            return []

    def _validate_and_merge_lesions(self, all_lesions: List[Dict], image: np.ndarray) -> List[Dict]:
        """Validate and merge overlapping lesions from different detection methods"""
        try:
            if not all_lesions:
                return []

            # Remove duplicates and merge overlapping detections
            merged_lesions = []

            for lesion in all_lesions:
                bbox = lesion['bbox']
                x, y, w, h = bbox

                # Check if this lesion overlaps significantly with existing ones
                is_duplicate = False
                for existing in merged_lesions:
                    ex, ey, ew, eh = existing['bbox']

                    # Calculate overlap
                    overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
                    overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
                    overlap_area = overlap_x * overlap_y

                    # Calculate overlap ratio
                    lesion_area = w * h
                    existing_area = ew * eh
                    overlap_ratio = overlap_area / min(lesion_area, existing_area)

                    if overlap_ratio > 0.5:  # 50% overlap threshold
                        is_duplicate = True
                        # Keep the larger lesion
                        if lesion_area > existing_area:
                            merged_lesions.remove(existing)
                            merged_lesions.append(lesion)
                        break

                if not is_duplicate:
                    merged_lesions.append(lesion)

            # Validate lesions based on medical criteria
            validated_lesions = []
            for lesion in merged_lesions:
                if self._validate_lesion_medical_criteria(lesion, image):
                    validated_lesions.append(lesion)

            return validated_lesions

        except Exception as e:
            logger.warning(f"⚠️ Lesion validation failed: {e}")
            return all_lesions[:10]  # Return first 10 if validation fails

    def _validate_lesion_medical_criteria(self, lesion: Dict, image: np.ndarray) -> bool:
        """Validate lesion based on medical criteria"""
        try:
            bbox = lesion['bbox']
            x, y, w, h = bbox
            area = lesion.get('area', w * h)

            # Size criteria (2mm to 50mm assuming 300 DPI)
            min_area = 50   # ~2mm diameter
            max_area = 50000  # ~50mm diameter

            if not (min_area <= area <= max_area):
                return False

            # Aspect ratio criteria (not too elongated)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 5.0:  # Too elongated
                return False

            # Color variation criteria
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                color_std = np.std(roi)
                if color_std < 5:  # Too uniform (likely artifact)
                    return False

            return True

        except Exception as e:
            logger.warning(f"⚠️ Medical validation failed: {e}")
            return True  # Default to valid if validation fails

    def _analyze_lesion_properties(self, lesion: Dict, image: np.ndarray, lesion_id: int) -> Dict[str, Any]:
        """Analyze detailed properties of detected lesion"""
        try:
            bbox = lesion['bbox']
            x, y, w, h = bbox

            # Extract lesion region
            lesion_roi = image[y:y+h, x:x+w]

            # Basic properties
            area = lesion.get('area', w * h)
            perimeter = lesion.get('perimeter', 2 * (w + h))  # Approximation

            # Shape analysis
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            aspect_ratio = max(w, h) / min(w, h)

            # Color analysis
            mean_color = np.mean(lesion_roi, axis=(0, 1)) if lesion_roi.size > 0 else [0, 0, 0]
            color_std = np.std(lesion_roi, axis=(0, 1)) if lesion_roi.size > 0 else [0, 0, 0]

            # Convert to HSV for better color analysis
            if lesion_roi.size > 0:
                hsv_roi = cv2.cvtColor(lesion_roi, cv2.COLOR_RGB2HSV)
                mean_hsv = np.mean(hsv_roi, axis=(0, 1))
            else:
                mean_hsv = [0, 0, 0]

            # Texture analysis
            if lesion_roi.size > 0:
                gray_roi = cv2.cvtColor(lesion_roi, cv2.COLOR_RGB2GRAY)
                texture_variance = np.var(gray_roi)
            else:
                texture_variance = 0

            # Medical risk assessment based on ABCDE criteria
            risk_factors = self._assess_abcde_criteria(lesion_roi, area, circularity, mean_color, color_std)

            return {
                'lesion_id': lesion_id,
                'bbox': bbox,
                'center': (x + w//2, y + h//2),
                'dimensions': {'width': w, 'height': h},
                'area_pixels': area,
                'estimated_size_mm': self._pixels_to_mm(area),
                'shape_properties': {
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'perimeter': perimeter
                },
                'color_properties': {
                    'mean_rgb': mean_color.tolist() if hasattr(mean_color, 'tolist') else mean_color,
                    'color_std': color_std.tolist() if hasattr(color_std, 'tolist') else color_std,
                    'mean_hsv': mean_hsv.tolist() if hasattr(mean_hsv, 'tolist') else mean_hsv
                },
                'texture_properties': {
                    'variance': texture_variance
                },
                'detection_method': lesion.get('detection_method', 'unknown'),
                'medical_assessment': risk_factors,
                'confidence_score': self._calculate_confidence_score(lesion, risk_factors)
            }

        except Exception as e:
            logger.warning(f"⚠️ Lesion analysis failed: {e}")
            return {
                'lesion_id': lesion_id,
                'bbox': bbox,
                'error': str(e)
            }

    def _assess_abcde_criteria(self, lesion_roi: np.ndarray, area: float,
                              circularity: float, mean_color: np.ndarray,
                              color_std: np.ndarray) -> Dict[str, Any]:
        """Assess lesion using ABCDE criteria for melanoma detection"""
        try:
            risk_factors = {
                'asymmetry': 'low',
                'border_irregularity': 'low',
                'color_variation': 'low',
                'diameter': 'low',
                'evolution': 'unknown',  # Cannot assess without temporal data
                'overall_risk': 'low'
            }

            # A - Asymmetry (based on circularity)
            if circularity < 0.6:
                risk_factors['asymmetry'] = 'high'
            elif circularity < 0.8:
                risk_factors['asymmetry'] = 'medium'

            # B - Border irregularity (based on circularity and shape)
            if circularity < 0.5:
                risk_factors['border_irregularity'] = 'high'
            elif circularity < 0.7:
                risk_factors['border_irregularity'] = 'medium'

            # C - Color variation (based on color standard deviation)
            if hasattr(color_std, '__len__') and len(color_std) >= 3:
                avg_color_std = np.mean(color_std)
                if avg_color_std > 30:
                    risk_factors['color_variation'] = 'high'
                elif avg_color_std > 15:
                    risk_factors['color_variation'] = 'medium'

            # D - Diameter (>6mm is concerning)
            estimated_diameter_mm = self._pixels_to_mm(area, assume_circular=True)
            if estimated_diameter_mm > 6:
                risk_factors['diameter'] = 'high'
            elif estimated_diameter_mm > 4:
                risk_factors['diameter'] = 'medium'

            # Overall risk assessment
            high_risk_count = sum(1 for factor in ['asymmetry', 'border_irregularity', 'color_variation', 'diameter']
                                if risk_factors[factor] == 'high')
            medium_risk_count = sum(1 for factor in ['asymmetry', 'border_irregularity', 'color_variation', 'diameter']
                                  if risk_factors[factor] == 'medium')

            if high_risk_count >= 2:
                risk_factors['overall_risk'] = 'high'
            elif high_risk_count >= 1 or medium_risk_count >= 3:
                risk_factors['overall_risk'] = 'medium'

            return risk_factors

        except Exception as e:
            logger.warning(f"⚠️ ABCDE assessment failed: {e}")
            return {'overall_risk': 'unknown', 'error': str(e)}

    def _pixels_to_mm(self, area_pixels: float, assume_circular: bool = False, dpi: float = 300) -> float:
        """Convert pixel area to estimated millimeter measurement"""
        try:
            # Assume 300 DPI for typical dermatological images
            pixels_per_mm = dpi / 25.4  # 25.4 mm per inch

            if assume_circular:
                # Calculate diameter for circular lesion
                radius_pixels = np.sqrt(area_pixels / np.pi)
                diameter_mm = (radius_pixels * 2) / pixels_per_mm
                return diameter_mm
            else:
                # Calculate equivalent diameter
                area_mm2 = area_pixels / (pixels_per_mm ** 2)
                diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)
                return diameter_mm

        except Exception as e:
            logger.warning(f"⚠️ Size conversion failed: {e}")
            return 0.0

    def _calculate_confidence_score(self, lesion: Dict, risk_factors: Dict) -> float:
        """Calculate confidence score for lesion detection"""
        try:
            base_confidence = 0.7  # Base confidence

            # Adjust based on detection method
            method = lesion.get('detection_method', 'unknown')
            if method == 'color_segmentation':
                base_confidence += 0.1
            elif method == 'texture_analysis':
                base_confidence += 0.05

            # Adjust based on size
            area = lesion.get('area', 0)
            if 500 < area < 10000:  # Optimal size range
                base_confidence += 0.1
            elif area < 100 or area > 30000:  # Too small or too large
                base_confidence -= 0.2

            # Adjust based on risk assessment
            overall_risk = risk_factors.get('overall_risk', 'low')
            if overall_risk == 'high':
                base_confidence += 0.1
            elif overall_risk == 'unknown':
                base_confidence -= 0.1

            return max(0.0, min(1.0, base_confidence))

        except Exception as e:
            logger.warning(f"⚠️ Confidence calculation failed: {e}")
            return 0.5

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'success': False,
            'error': error_message,
            'lesions_detected': 0,
            'lesions': [],
            'timestamp': time.time()
        }
