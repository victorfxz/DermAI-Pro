#!/usr/bin/env python3
"""
DermAI Pro - Responsive Design Configuration
Configuration for responsive UI elements and breakpoints

🎨 Responsive design system
📱 Multi-resolution support
🖥️ Adaptive layouts
"""

class ResponsiveConfig:
    """Configuration class for responsive design elements"""
    
    # Screen breakpoints (width in pixels)
    BREAKPOINTS = {
        'xs': 0,      # Extra small (mobile)
        'sm': 768,    # Small (tablet)
        'md': 1024,   # Medium (laptop)
        'lg': 1440,   # Large (desktop)
        'xl': 1920,   # Extra large (large desktop)
        'xxl': 2560   # Ultra large (4K)
    }
    
    # Font size configurations
    FONT_SIZES = {
        'xs': {
            'header_title': 18,
            'header_subtitle': 10,
            'section_title': 12,
            'button_text': 10,
            'body_text': 9,
            'small_text': 8
        },
        'sm': {
            'header_title': 22,
            'header_subtitle': 12,
            'section_title': 14,
            'button_text': 11,
            'body_text': 10,
            'small_text': 9
        },
        'md': {
            'header_title': 26,
            'header_subtitle': 14,
            'section_title': 16,
            'button_text': 12,
            'body_text': 11,
            'small_text': 10
        },
        'lg': {
            'header_title': 30,
            'header_subtitle': 16,
            'section_title': 18,
            'button_text': 14,
            'body_text': 12,
            'small_text': 11
        },
        'xl': {
            'header_title': 34,
            'header_subtitle': 18,
            'section_title': 20,
            'button_text': 16,
            'body_text': 14,
            'small_text': 12
        },
        'xxl': {
            'header_title': 38,
            'header_subtitle': 20,
            'section_title': 22,
            'button_text': 18,
            'body_text': 16,
            'small_text': 14
        }
    }
    
    # Spacing configurations
    SPACING = {
        'xs': {
            'padding_small': 5,
            'padding_medium': 8,
            'padding_large': 12,
            'margin_small': 3,
            'margin_medium': 6,
            'margin_large': 10
        },
        'sm': {
            'padding_small': 8,
            'padding_medium': 12,
            'padding_large': 16,
            'margin_small': 5,
            'margin_medium': 8,
            'margin_large': 12
        },
        'md': {
            'padding_small': 10,
            'padding_medium': 15,
            'padding_large': 20,
            'margin_small': 6,
            'margin_medium': 10,
            'margin_large': 15
        },
        'lg': {
            'padding_small': 12,
            'padding_medium': 18,
            'padding_large': 25,
            'margin_small': 8,
            'margin_medium': 12,
            'margin_large': 18
        },
        'xl': {
            'padding_small': 15,
            'padding_medium': 22,
            'padding_large': 30,
            'margin_small': 10,
            'margin_medium': 15,
            'margin_large': 22
        },
        'xxl': {
            'padding_small': 18,
            'padding_medium': 26,
            'padding_large': 35,
            'margin_small': 12,
            'margin_medium': 18,
            'margin_large': 26
        }
    }
    
    # Component sizes
    COMPONENT_SIZES = {
        'xs': {
            'button_height': 30,
            'header_height': 50,
            'panel_min_width': 180,
            'image_min_size': 200
        },
        'sm': {
            'button_height': 35,
            'header_height': 60,
            'panel_min_width': 220,
            'image_min_size': 250
        },
        'md': {
            'button_height': 40,
            'header_height': 70,
            'panel_min_width': 250,
            'image_min_size': 300
        },
        'lg': {
            'button_height': 45,
            'header_height': 80,
            'panel_min_width': 300,
            'image_min_size': 400
        },
        'xl': {
            'button_height': 50,
            'header_height': 90,
            'panel_min_width': 350,
            'image_min_size': 500
        },
        'xxl': {
            'button_height': 55,
            'header_height': 100,
            'panel_min_width': 400,
            'image_min_size': 600
        }
    }
    
    # Panel width ratios (as percentage of total width)
    PANEL_RATIOS = {
        'xs': {
            'left_panel': 0.35,
            'center_panel': 0.65,
            'right_panel': 0.0,  # Hidden on extra small screens
            'layout': 'stacked'
        },
        'sm': {
            'left_panel': 0.3,
            'center_panel': 0.7,
            'right_panel': 0.0,  # Hidden on small screens
            'layout': 'two_column'
        },
        'md': {
            'left_panel': 0.25,
            'center_panel': 0.45,
            'right_panel': 0.3,
            'layout': 'three_column'
        },
        'lg': {
            'left_panel': 0.22,
            'center_panel': 0.48,
            'right_panel': 0.3,
            'layout': 'three_column'
        },
        'xl': {
            'left_panel': 0.2,
            'center_panel': 0.5,
            'right_panel': 0.3,
            'layout': 'three_column'
        },
        'xxl': {
            'left_panel': 0.18,
            'center_panel': 0.52,
            'right_panel': 0.3,
            'layout': 'three_column'
        }
    }
    
    # Text content for different screen sizes
    TEXT_CONTENT = {
        'xs': {
            'header_title': '🏥 DermAI',
            'header_subtitle': 'AI Dermatology',
            'load_button': '📁 Load',
            'analyze_button': '🔬 Analyze',
            'workspace_title': '🖼️ IMAGE',
            'results_title': '📊 RESULTS'
        },
        'sm': {
            'header_title': '🏥 DermAI Pro',
            'header_subtitle': 'Professional AI System',
            'load_button': '📁 Load Image',
            'analyze_button': '🔬 Start Analysis',
            'workspace_title': '🖼️ WORKSPACE',
            'results_title': '📊 RESULTS'
        },
        'md': {
            'header_title': '🏥 DermAI Pro',
            'header_subtitle': 'Professional Dermatology AI System',
            'load_button': '📁 Load Dermatological Image',
            'analyze_button': '🔬 Start AI Analysis',
            'workspace_title': '🖼️ IMAGE WORKSPACE',
            'results_title': '📊 ANALYSIS RESULTS'
        },
        'lg': {
            'header_title': '🏥 DermAI Pro',
            'header_subtitle': 'Professional Dermatology AI System • Powered by Gemma 3n-E4B',
            'load_button': '📁 Load Dermatological Image',
            'analyze_button': '🔬 Start AI Analysis',
            'workspace_title': '🖼️ IMAGE ANALYSIS WORKSPACE',
            'results_title': '📊 ANALYSIS RESULTS'
        },
        'xl': {
            'header_title': '🏥 DermAI Pro',
            'header_subtitle': 'Professional Dermatology AI System • Powered by Gemma 3n-E4B',
            'load_button': '📁 Load Dermatological Image',
            'analyze_button': '🔬 Start AI Analysis',
            'workspace_title': '🖼️ IMAGE ANALYSIS WORKSPACE',
            'results_title': '📊 ANALYSIS RESULTS'
        },
        'xxl': {
            'header_title': '🏥 DermAI Pro',
            'header_subtitle': 'Professional Dermatology AI System • Powered by Gemma 3n-E4B',
            'load_button': '📁 Load Dermatological Image',
            'analyze_button': '🔬 Start AI Analysis',
            'workspace_title': '🖼️ IMAGE ANALYSIS WORKSPACE',
            'results_title': '📊 ANALYSIS RESULTS'
        }
    }
    
    @classmethod
    def get_breakpoint(cls, width):
        """Get current breakpoint based on width"""
        if width < cls.BREAKPOINTS['sm']:
            return 'xs'
        elif width < cls.BREAKPOINTS['md']:
            return 'sm'
        elif width < cls.BREAKPOINTS['lg']:
            return 'md'
        elif width < cls.BREAKPOINTS['xl']:
            return 'lg'
        elif width < cls.BREAKPOINTS['xxl']:
            return 'xl'
        else:
            return 'xxl'
    
    @classmethod
    def get_config(cls, width, config_type):
        """Get configuration for specific breakpoint and type"""
        breakpoint = cls.get_breakpoint(width)
        
        config_map = {
            'fonts': cls.FONT_SIZES,
            'spacing': cls.SPACING,
            'components': cls.COMPONENT_SIZES,
            'panels': cls.PANEL_RATIOS,
            'text': cls.TEXT_CONTENT
        }
        
        return config_map.get(config_type, {}).get(breakpoint, {})
    
    @classmethod
    def get_responsive_value(cls, width, config_type, key, default=None):
        """Get specific responsive value"""
        config = cls.get_config(width, config_type)
        return config.get(key, default)
