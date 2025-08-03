#!/usr/bin/env python3
"""
DermAI Pro - AI Analysis Engine
Professional dermatological AI analysis using Gemma 3n-E4B

🤖 Real AI model integration via Ollama
🔬 Medical-grade analysis for 14+ skin conditions
🏥 Professional diagnostic assistance
"""

import logging
import time
import base64
import io
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
import requests
from PIL import Image

logger = logging.getLogger("DermAI-Pro.AI-Engine")

class DermAIEngine:
    """
    Professional Dermatological AI Analysis Engine
    
    Uses Gemma 3n-E4B model via Ollama for medical-grade
    dermatological analysis and diagnosis assistance.
    """
    
    def __init__(self):
        """Initialize the DermAI analysis engine"""
        self.model_name = "gemma3n:e4b"
        self.ollama_url = "http://localhost:11434"
        self.initialized = False
        self.model_ready = False
        
        # Medical conditions database
        self.conditions_database = {
            'melanoma': {
                'name': 'Melanoma',
                'category': 'malignant',
                'urgency': 'high',
                'description': 'Most dangerous form of skin cancer'
            },
            'basal_cell_carcinoma': {
                'name': 'Basal Cell Carcinoma',
                'category': 'malignant',
                'urgency': 'medium',
                'description': 'Most common form of skin cancer'
            },
            'squamous_cell_carcinoma': {
                'name': 'Squamous Cell Carcinoma',
                'category': 'malignant',
                'urgency': 'high',
                'description': 'Aggressive form of skin cancer'
            },
            'actinic_keratosis': {
                'name': 'Actinic Keratosis',
                'category': 'pre_malignant',
                'urgency': 'medium',
                'description': 'Pre-cancerous lesions from sun damage'
            },
            'melanocytic_nevus': {
                'name': 'Melanocytic Nevus',
                'category': 'benign',
                'urgency': 'low',
                'description': 'Common moles and pigmented lesions'
            },
            'seborrheic_keratosis': {
                'name': 'Seborrheic Keratosis',
                'category': 'benign',
                'urgency': 'low',
                'description': 'Benign age-related skin growths'
            },
            'dermatofibroma': {
                'name': 'Dermatofibroma',
                'category': 'benign',
                'urgency': 'low',
                'description': 'Benign fibrous skin nodules'
            },
            'vascular_lesion': {
                'name': 'Vascular Lesion',
                'category': 'benign',
                'urgency': 'low',
                'description': 'Blood vessel related skin lesions'
            },
            'monkeypox': {
                'name': 'Monkeypox',
                'category': 'infectious',
                'urgency': 'high',
                'description': 'Viral infection with skin manifestations'
            },
            'chickenpox': {
                'name': 'Chickenpox',
                'category': 'infectious',
                'urgency': 'medium',
                'description': 'Varicella-zoster virus infection'
            },
            'measles': {
                'name': 'Measles',
                'category': 'infectious',
                'urgency': 'high',
                'description': 'Highly contagious viral infection'
            },
            'hand_foot_mouth': {
                'name': 'Hand, Foot, and Mouth Disease',
                'category': 'infectious',
                'urgency': 'medium',
                'description': 'Viral infection common in children'
            },
            'cowpox': {
                'name': 'Cowpox',
                'category': 'infectious',
                'urgency': 'medium',
                'description': 'Viral infection from animal contact'
            },
            'healthy_skin': {
                'name': 'Healthy Skin',
                'category': 'normal',
                'urgency': 'low',
                'description': 'Normal, healthy skin appearance'
            }
        }
        
        logger.info("🤖 DermAI Engine initialized - ready for medical analysis")
    
    def initialize_model(self) -> bool:
        """Initialize and test the AI model connection"""
        try:
            logger.info("Initializing Gemma 3n-E4B model connection...")

            # Test Ollama connection with longer timeout
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=15)
            if response.status_code != 200:
                logger.error("Ollama server not responding")
                return False

            # Check if model is available
            models = response.json().get('models', [])
            model_available = any(self.model_name in model.get('name', '') for model in models)

            if not model_available:
                logger.error(f"Model {self.model_name} not found in Ollama")
                logger.info("Available models:")
                for model in models:
                    logger.info(f"  - {model.get('name', 'Unknown')}")
                return False

            # Test model with simple prompt and longer timeout
            logger.info("Testing model with simple prompt...")
            test_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": "What is dermatology? Answer in one sentence.",
                    "stream": False,
                    "options": {
                        "num_predict": 30,
                        "temperature": 0.7
                    }
                },
                timeout=120  # 2 minutes timeout for model test
            )

            if test_response.status_code == 200:
                test_result = test_response.json()
                if 'response' in test_result and len(test_result['response']) > 5:
                    logger.info("Model test successful - ready for medical analysis")
                    logger.info(f"Test response: {test_result['response'][:100]}...")
                    self.initialized = True
                    self.model_ready = True
                    return True
                else:
                    logger.error("Model test returned empty response")
                    return False
            else:
                logger.error(f"Model test failed with status: {test_response.status_code}")
                return False

        except requests.exceptions.Timeout as e:
            logger.error(f"Model initialization timed out: {e}")
            logger.error("This usually means the model is loading. Please wait and try again.")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to Ollama server: {e}")
            logger.error("Please ensure Ollama is running: ollama serve")
            return False
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string for Ollama"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"❌ Image conversion failed: {e}")
            return ""
    
    def _create_medical_prompt(self, analysis_type: str = "multi", target_condition: str = None) -> str:
        """Create professional medical analysis prompt"""
        
        base_prompt = """You are a professional dermatology AI assistant analyzing a skin lesion image. 
Provide a detailed medical analysis following these guidelines:

ANALYSIS REQUIREMENTS:
- Examine the image carefully for any skin lesions, discoloration, or abnormalities
- Consider morphological features: size, shape, color, texture, borders
- Evaluate for signs of malignancy using clinical criteria
- Assess for infectious conditions if relevant

"""
        
        if analysis_type == "multi":
            conditions_list = "\n".join([f"- {info['name']}: {info['description']}" 
                                       for info in self.conditions_database.values()])
            
            prompt = base_prompt + f"""
MULTI-CONDITION ANALYSIS:
Analyze for these dermatological conditions:
{conditions_list}

For each relevant condition, provide:
CONDITION: [Condition Name]
PROBABILITY: [0-100%]
CONFIDENCE: [0-100%]
KEY FEATURES: [Observed features supporting this diagnosis]
CLINICAL ASSESSMENT: [Brief clinical interpretation]
RISK LEVEL: [low/medium/high]

Focus on conditions with probability >5%. Provide detailed analysis for top 3-5 most likely conditions.
"""
        
        else:  # single condition
            condition_info = self.conditions_database.get(target_condition.lower().replace(' ', '_'), {})
            condition_name = condition_info.get('name', target_condition)
            
            prompt = base_prompt + f"""
SINGLE-CONDITION ANALYSIS for {condition_name}:
Focus specifically on evaluating for {condition_name}.

Provide detailed analysis:
CONDITION: {condition_name}
PROBABILITY: [0-100%]
CONFIDENCE: [0-100%]
KEY FEATURES PRESENT: [Features supporting this diagnosis]
KEY FEATURES ABSENT: [Expected features that are missing]
CLINICAL ASSESSMENT: [Detailed clinical interpretation]
DIFFERENTIAL DIAGNOSIS: [Alternative conditions to consider]
RECOMMENDATIONS: [Clinical recommendations]
RISK LEVEL: [low/medium/high]
"""
        
        prompt += """
IMPORTANT: Base analysis only on visible features in the image. 
Be precise with percentages and provide clinical reasoning for all assessments.
"""
        
        return prompt

    def analyze_image(self, image: np.ndarray, analysis_type: str = "multi",
                     target_condition: str = None) -> Dict[str, Any]:
        """
        Analyze dermatological image using AI model

        Args:
            image: Input image as numpy array
            analysis_type: "multi" for all conditions, "single" for specific condition
            target_condition: Specific condition name for single analysis

        Returns:
            Dictionary with analysis results
        """
        try:
            if not self.model_ready:
                if not self.initialize_model():
                    return {
                        'success': False,
                        'error': 'AI model not available',
                        'timestamp': time.time()
                    }

            logger.info(f"🔬 Starting {analysis_type} dermatological analysis...")
            start_time = time.time()

            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            if not image_base64:
                return {
                    'success': False,
                    'error': 'Image conversion failed',
                    'timestamp': time.time()
                }

            # Create medical prompt
            prompt = self._create_medical_prompt(analysis_type, target_condition)

            # Send request to Ollama
            logger.info("Sending image to AI model for analysis...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "num_predict": 1000
                    }
                },
                timeout=600  # 10 minutes timeout for image analysis
            )

            if response.status_code != 200:
                logger.error(f"❌ AI model request failed: {response.status_code}")
                return {
                    'success': False,
                    'error': f'Model request failed: {response.status_code}',
                    'timestamp': time.time()
                }

            result = response.json()
            ai_response = result.get('response', '')
            processing_time = time.time() - start_time

            logger.info(f"✅ AI analysis completed in {processing_time:.2f}s")
            logger.info(f"📝 Response length: {len(ai_response)} characters")

            # Parse the AI response
            if analysis_type == "multi":
                parsed_results = self._parse_multi_condition_response(ai_response)
            else:
                parsed_results = self._parse_single_condition_response(ai_response, target_condition)

            # Combine results
            final_results = {
                'success': True,
                'analysis_type': analysis_type,
                'target_condition': target_condition,
                'raw_response': ai_response,
                'processing_time': processing_time,
                'model_used': self.model_name,
                'timestamp': time.time(),
                **parsed_results
            }

            return final_results

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

    def _parse_multi_condition_response(self, response: str) -> Dict[str, Any]:
        """Parse multi-condition analysis response"""
        try:
            conditions_detected = {}
            lines = response.strip().split('\n')
            current_condition = None
            current_data = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for condition headers
                if line.upper().startswith('CONDITION:'):
                    # Save previous condition
                    if current_condition and current_data:
                        conditions_detected[current_condition] = current_data.copy()

                    # Start new condition
                    condition_name = line.split(':', 1)[1].strip()
                    current_condition = condition_name.lower().replace(' ', '_').replace('-', '_')
                    current_data = {
                        'name': condition_name,
                        'probability': 0.0,
                        'confidence': 0.0,
                        'key_features': [],
                        'clinical_assessment': '',
                        'risk_level': 'low'
                    }

                # Parse probability
                elif line.upper().startswith('PROBABILITY:') and current_condition:
                    prob_text = line.split(':', 1)[1].strip()
                    try:
                        import re
                        prob_match = re.search(r'(\d+\.?\d*)', prob_text)
                        if prob_match:
                            prob = float(prob_match.group(1))
                            if prob > 1.0:  # Convert percentage to decimal
                                prob = prob / 100.0
                            current_data['probability'] = max(0.0, min(1.0, prob))
                    except:
                        pass

                # Parse confidence
                elif line.upper().startswith('CONFIDENCE:') and current_condition:
                    conf_text = line.split(':', 1)[1].strip()
                    try:
                        import re
                        conf_match = re.search(r'(\d+\.?\d*)', conf_text)
                        if conf_match:
                            conf = float(conf_match.group(1))
                            if conf > 1.0:  # Convert percentage to decimal
                                conf = conf / 100.0
                            current_data['confidence'] = max(0.0, min(1.0, conf))
                    except:
                        pass

                # Parse risk level
                elif line.upper().startswith('RISK LEVEL:') and current_condition:
                    risk_text = line.split(':', 1)[1].strip().lower()
                    if 'high' in risk_text:
                        current_data['risk_level'] = 'high'
                    elif 'medium' in risk_text:
                        current_data['risk_level'] = 'medium'
                    else:
                        current_data['risk_level'] = 'low'

                # Parse key features
                elif line.upper().startswith('KEY FEATURES:') and current_condition:
                    features_text = line.split(':', 1)[1].strip()
                    if features_text:
                        features = [f.strip() for f in features_text.split(',')]
                        current_data['key_features'] = [f for f in features if f and len(f) > 2]

                # Parse clinical assessment
                elif line.upper().startswith('CLINICAL ASSESSMENT:') and current_condition:
                    assessment = line.split(':', 1)[1].strip()
                    if assessment:
                        current_data['clinical_assessment'] = assessment

            # Save last condition
            if current_condition and current_data:
                conditions_detected[current_condition] = current_data

            # Generate overall assessment
            total_conditions = len(conditions_detected)
            high_risk_count = sum(1 for c in conditions_detected.values() if c.get('risk_level') == 'high')

            return {
                'conditions_detected': conditions_detected,
                'total_conditions_analyzed': total_conditions,
                'high_risk_conditions': high_risk_count,
                'overall_risk': 'high' if high_risk_count > 0 else 'low'
            }

        except Exception as e:
            logger.error(f"❌ Multi-condition parsing failed: {e}")
            return {
                'conditions_detected': {},
                'parsing_error': str(e)
            }

    def _parse_single_condition_response(self, response: str, target_condition: str) -> Dict[str, Any]:
        """Parse single condition analysis response"""
        try:
            result = {
                'condition_name': target_condition,
                'probability': 0.0,
                'confidence': 0.0,
                'key_features_present': [],
                'key_features_absent': [],
                'clinical_assessment': '',
                'differential_diagnosis': '',
                'recommendations': [],
                'risk_level': 'low'
            }

            lines = response.strip().split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Parse probability
                if line.upper().startswith('PROBABILITY:'):
                    prob_text = line.split(':', 1)[1].strip()
                    try:
                        import re
                        prob_match = re.search(r'(\d+\.?\d*)', prob_text)
                        if prob_match:
                            prob = float(prob_match.group(1))
                            if prob > 1.0:  # Convert percentage to decimal
                                prob = prob / 100.0
                            result['probability'] = max(0.0, min(1.0, prob))
                    except:
                        pass

                # Parse confidence
                elif line.upper().startswith('CONFIDENCE:'):
                    conf_text = line.split(':', 1)[1].strip()
                    try:
                        import re
                        conf_match = re.search(r'(\d+\.?\d*)', conf_text)
                        if conf_match:
                            conf = float(conf_match.group(1))
                            if conf > 1.0:  # Convert percentage to decimal
                                conf = conf / 100.0
                            result['confidence'] = max(0.0, min(1.0, conf))
                    except:
                        pass

                # Parse risk level
                elif line.upper().startswith('RISK LEVEL:'):
                    risk_text = line.split(':', 1)[1].strip().lower()
                    if 'high' in risk_text:
                        result['risk_level'] = 'high'
                    elif 'medium' in risk_text:
                        result['risk_level'] = 'medium'
                    else:
                        result['risk_level'] = 'low'

                # Parse key features present
                elif line.upper().startswith('KEY FEATURES PRESENT:'):
                    features_text = line.split(':', 1)[1].strip()
                    if features_text:
                        features = [f.strip() for f in features_text.split(',')]
                        result['key_features_present'] = [f for f in features if f and len(f) > 2]

                # Parse key features absent
                elif line.upper().startswith('KEY FEATURES ABSENT:'):
                    features_text = line.split(':', 1)[1].strip()
                    if features_text:
                        features = [f.strip() for f in features_text.split(',')]
                        result['key_features_absent'] = [f for f in features if f and len(f) > 2]

                # Parse clinical assessment
                elif line.upper().startswith('CLINICAL ASSESSMENT:'):
                    assessment = line.split(':', 1)[1].strip()
                    if assessment:
                        result['clinical_assessment'] = assessment

                # Parse differential diagnosis
                elif line.upper().startswith('DIFFERENTIAL DIAGNOSIS:'):
                    differential = line.split(':', 1)[1].strip()
                    if differential:
                        result['differential_diagnosis'] = differential

                # Parse recommendations
                elif line.upper().startswith('RECOMMENDATIONS:'):
                    rec_text = line.split(':', 1)[1].strip()
                    if rec_text:
                        recommendations = [r.strip() for r in rec_text.split(',')]
                        result['recommendations'] = [r for r in recommendations if r and len(r) > 5]

            return {
                'single_condition_analysis': result,
                'overall_risk': result['risk_level']
            }

        except Exception as e:
            logger.error(f"❌ Single condition parsing failed: {e}")
            return {
                'single_condition_analysis': {
                    'condition_name': target_condition,
                    'probability': 0.0,
                    'confidence': 0.0,
                    'error': str(e)
                },
                'parsing_error': str(e)
            }

    def get_clinical_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on analysis results"""
        try:
            recommendations = []

            if analysis_results.get('analysis_type') == 'multi':
                conditions = analysis_results.get('conditions_detected', {})
                high_risk_found = False

                for condition_id, condition_data in conditions.items():
                    probability = condition_data.get('probability', 0.0)
                    risk_level = condition_data.get('risk_level', 'low')
                    condition_name = condition_data.get('name', condition_id)

                    if probability > 0.3 and risk_level == 'high':  # High probability + high risk
                        recommendations.append(f"Urgent dermatologist consultation for suspected {condition_name}")
                        high_risk_found = True
                    elif probability > 0.5:  # High probability
                        recommendations.append(f"Professional evaluation recommended for {condition_name}")

                if not high_risk_found:
                    recommendations.extend([
                        "Routine dermatological examination recommended",
                        "Monitor lesion for changes in size, color, or texture",
                        "Use broad-spectrum sunscreen (SPF 30+) daily",
                        "Perform regular self-skin examinations"
                    ])
                else:
                    recommendations.extend([
                        "Document lesion characteristics with photography",
                        "Avoid sun exposure until professional evaluation",
                        "Schedule follow-up within recommended timeframe"
                    ])

            else:  # Single condition
                single_analysis = analysis_results.get('single_condition_analysis', {})
                probability = single_analysis.get('probability', 0.0)
                risk_level = single_analysis.get('risk_level', 'low')
                condition_name = single_analysis.get('condition_name', 'Unknown')

                # Use AI-generated recommendations if available
                ai_recommendations = single_analysis.get('recommendations', [])
                if ai_recommendations:
                    recommendations.extend(ai_recommendations)

                # Add standard recommendations based on risk
                if probability > 0.5 and risk_level == 'high':
                    recommendations.append(f"Urgent evaluation for {condition_name} recommended")
                elif probability > 0.3:
                    recommendations.append(f"Professional assessment for {condition_name} advised")
                else:
                    recommendations.append("Routine monitoring recommended")

            # Ensure we have at least basic recommendations
            if not recommendations:
                recommendations = [
                    "Continue routine skin care and monitoring",
                    "Use sun protection measures",
                    "Consult healthcare provider if concerns arise"
                ]

            return recommendations[:6]  # Limit to 6 recommendations

        except Exception as e:
            logger.error(f"❌ Recommendation generation failed: {e}")
            return [
                "Professional dermatological evaluation recommended",
                "Monitor skin changes regularly",
                "Use appropriate sun protection"
            ]
