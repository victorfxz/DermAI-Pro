#!/usr/bin/env python3
"""
DermAI Pro - Main Professional Interface
Modern medical-grade GUI for dermatological AI analysis

🏥 Professional medical interface design
🎯 Optimized for clinical workflow
🔬 Real-time AI analysis integration
"""

import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import numpy as np
import cv2
from PIL import Image, ImageTk

# Import our core modules
from ..core.ai_engine import DermAIEngine
from ..core.lesion_detector import ProfessionalLesionDetector
from .responsive_config import ResponsiveConfig

logger = logging.getLogger("DermAI-Pro.Interface")

class DermAIProInterface:
    """
    Professional DermAI Pro Main Interface
    
    Modern, medical-grade GUI for dermatological AI analysis
    designed for professional clinical use.
    """
    
    def __init__(self):
        """Initialize the professional interface"""
        self.root = None
        self.current_image = None
        self.current_image_path = None
        self.analysis_results = None
        self.processing = False
        
        # Initialize AI components
        self.ai_engine = DermAIEngine()
        self.lesion_detector = ProfessionalLesionDetector()
        
        # UI Colors - Professional Medical Theme
        self.colors = {
            'primary': '#2E86AB',      # Medical blue
            'secondary': '#A23B72',    # Accent purple
            'success': '#28A745',      # Success green
            'warning': '#FFC107',      # Warning yellow
            'danger': '#DC3545',       # Danger red
            'background': '#F8F9FA',   # Light background
            'surface': '#FFFFFF',      # White surface
            'text_primary': '#212529', # Dark text
            'text_secondary': '#6C757D', # Gray text
            'border': '#DEE2E6'        # Light border
        }
        
        logger.info("🏥 DermAI Pro Interface initialized")
    
    def run(self):
        """Launch the professional interface"""
        try:
            logger.info("Launching DermAI Pro professional interface...")

            # Create main window
            self.root = ctk.CTk()
            self.root.title("🏥 DermAI Pro - Professional Dermatology AI System v1.0.0")

            # Get screen dimensions for responsive sizing
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Calculate responsive window size (80% of screen)
            window_width = int(screen_width * 0.8)
            window_height = int(screen_height * 0.8)

            # Ensure minimum and maximum sizes
            window_width = max(1000, min(window_width, 1600))
            window_height = max(700, min(window_height, 1200))

            # Center window on screen
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2

            self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
            self.root.minsize(1000, 700)  # Responsive minimum size

            # Store window dimensions for responsive calculations
            self.window_width = window_width
            self.window_height = window_height

            # Initialize responsive configuration
            self.responsive_config = ResponsiveConfig()
            self.current_breakpoint = self.responsive_config.get_breakpoint(window_width)

            # Get responsive configurations
            self.fonts = self.responsive_config.get_config(window_width, 'fonts')
            self.spacing = self.responsive_config.get_config(window_width, 'spacing')
            self.components = self.responsive_config.get_config(window_width, 'components')
            self.panels = self.responsive_config.get_config(window_width, 'panels')
            self.text_content = self.responsive_config.get_config(window_width, 'text')

            # Set window icon (if available)
            try:
                icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.ico"
                if icon_path.exists():
                    self.root.iconbitmap(str(icon_path))
            except:
                pass

            # Configure responsive grid weights
            self.root.grid_columnconfigure(0, weight=0, minsize=250)  # Left panel
            self.root.grid_columnconfigure(1, weight=2, minsize=400)  # Center panel
            self.root.grid_columnconfigure(2, weight=1, minsize=300)  # Right panel
            self.root.grid_rowconfigure(0, weight=0)  # Header
            self.root.grid_rowconfigure(1, weight=1)  # Main content
            self.root.grid_rowconfigure(2, weight=0)  # Status bar

            # Bind resize event for dynamic responsiveness
            self.root.bind('<Configure>', self.on_window_resize)

            # Create interface components
            self.create_header()
            self.create_main_layout()
            self.create_status_bar()

            # Initialize AI engine
            self.initialize_ai_engine()

            # Start the application
            logger.info("DermAI Pro interface ready")
            self.root.mainloop()

        except Exception as e:
            logger.error(f"Interface launch failed: {e}")
            if self.root:
                messagebox.showerror("DermAI Pro Error", f"Interface error: {str(e)}")

    def on_window_resize(self, event):
        """Handle window resize events for responsive design"""
        try:
            # Only handle resize events for the main window
            if event.widget == self.root:
                new_width = event.width
                new_height = event.height

                # Update stored dimensions
                self.window_width = new_width
                self.window_height = new_height

                # Check if breakpoint changed
                new_breakpoint = self.responsive_config.get_breakpoint(new_width)
                if new_breakpoint != self.current_breakpoint:
                    self.current_breakpoint = new_breakpoint
                    self.update_responsive_configs()

                # Adjust layout based on new size
                self.adjust_responsive_layout()

        except Exception as e:
            logger.error(f"Window resize handling failed: {e}")

    def update_responsive_configs(self):
        """Update responsive configurations when breakpoint changes"""
        try:
            self.fonts = self.responsive_config.get_config(self.window_width, 'fonts')
            self.spacing = self.responsive_config.get_config(self.window_width, 'spacing')
            self.components = self.responsive_config.get_config(self.window_width, 'components')
            self.panels = self.responsive_config.get_config(self.window_width, 'panels')
            self.text_content = self.responsive_config.get_config(self.window_width, 'text')

            # Trigger full UI refresh for major breakpoint changes
            self.refresh_ui_elements()

        except Exception as e:
            logger.error(f"Responsive config update failed: {e}")

    def refresh_ui_elements(self):
        """Refresh UI elements with new responsive settings"""
        try:
            # Update text content
            if hasattr(self, 'upload_btn'):
                self.upload_btn.configure(text=self.text_content.get('load_button', '📁 Load Image'))

            if hasattr(self, 'analyze_btn'):
                self.analyze_btn.configure(text=self.text_content.get('analyze_button', '🔬 Analyze'))

            # Update image display if needed
            if hasattr(self, 'current_image') and self.current_image is not None:
                self.update_image_display_size()

        except Exception as e:
            logger.error(f"UI elements refresh failed: {e}")

    def adjust_responsive_layout(self):
        """Adjust layout elements based on current window size"""
        try:
            # Calculate responsive panel widths
            if self.window_width < 1200:
                # Compact layout for smaller screens
                left_width = max(200, int(self.window_width * 0.25))
                right_width = max(250, int(self.window_width * 0.35))
            else:
                # Standard layout for larger screens
                left_width = max(250, int(self.window_width * 0.2))
                right_width = max(300, int(self.window_width * 0.3))

            # Update grid column configurations
            self.root.grid_columnconfigure(0, minsize=left_width)
            self.root.grid_columnconfigure(2, minsize=right_width)

            # Adjust image display size based on available space
            if hasattr(self, 'current_image') and self.current_image is not None:
                self.update_image_display_size()

        except Exception as e:
            logger.error(f"Responsive layout adjustment failed: {e}")

    def update_image_display_size(self):
        """Update image display size based on current window dimensions"""
        try:
            # Calculate available space for image
            center_width = self.window_width - 550  # Account for left and right panels
            center_height = self.window_height - 200  # Account for header and status

            # Set responsive image display size
            self.display_width = max(300, min(center_width - 100, 600))
            self.display_height = max(250, min(center_height - 150, 500))

            # Redisplay current image if available
            if hasattr(self, 'current_image') and self.current_image is not None:
                self.display_image(self.current_image)

        except Exception as e:
            logger.error(f"Image display size update failed: {e}")

    def detect_screen_type(self):
        """Detect screen type for responsive design"""
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            # Detect screen type based on resolution
            if screen_width <= 1366 and screen_height <= 768:
                return "small"  # Laptop/small desktop
            elif screen_width <= 1920 and screen_height <= 1080:
                return "medium"  # Standard desktop
            elif screen_width <= 2560 and screen_height <= 1440:
                return "large"  # Large desktop/QHD
            else:
                return "xlarge"  # 4K and above

        except Exception as e:
            logger.error(f"Screen type detection failed: {e}")
            return "medium"  # Default fallback

    def setup_responsive_scaling(self):
        """Setup responsive scaling factors based on screen type"""
        try:
            scaling_factors = {
                "small": {
                    "font_scale": 0.85,
                    "padding_scale": 0.8,
                    "button_scale": 0.9,
                    "panel_scale": 0.9
                },
                "medium": {
                    "font_scale": 1.0,
                    "padding_scale": 1.0,
                    "button_scale": 1.0,
                    "panel_scale": 1.0
                },
                "large": {
                    "font_scale": 1.1,
                    "padding_scale": 1.2,
                    "button_scale": 1.1,
                    "panel_scale": 1.1
                },
                "xlarge": {
                    "font_scale": 1.3,
                    "padding_scale": 1.4,
                    "button_scale": 1.2,
                    "panel_scale": 1.2
                }
            }

            self.scale = scaling_factors.get(self.screen_type, scaling_factors["medium"])

        except Exception as e:
            logger.error(f"Responsive scaling setup failed: {e}")
            # Default scaling
            self.scale = {
                "font_scale": 1.0,
                "padding_scale": 1.0,
                "button_scale": 1.0,
                "panel_scale": 1.0
            }
    
    def create_header(self):
        """Create responsive professional header with branding"""
        # Calculate responsive header height
        header_height = max(60, min(int(self.window_height * 0.08), 100))

        header_frame = ctk.CTkFrame(self.root, height=header_height, fg_color=self.colors['primary'])
        header_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=0, pady=0)
        header_frame.grid_propagate(False)

        # Configure header grid for responsive layout
        header_frame.grid_columnconfigure(0, weight=0)  # Logo
        header_frame.grid_columnconfigure(1, weight=1)  # Title area
        header_frame.grid_columnconfigure(2, weight=0)  # Status

        # Logo and main title
        title_size = max(20, min(int(self.window_width / 50), 32))
        title_label = ctk.CTkLabel(
            header_frame,
            text="🏥 DermAI Pro",
            font=ctk.CTkFont(size=title_size, weight="bold"),
            text_color="white"
        )
        title_label.grid(row=0, column=0, padx=15, pady=10, sticky="w")

        # Responsive subtitle (hide on very small screens)
        if self.window_width > 1000:
            subtitle_size = max(11, min(int(self.window_width / 100), 16))
            subtitle_text = "Professional Dermatology AI System • Powered by Gemma 3n-E4B"

            # Shorten subtitle on medium screens
            if self.window_width < 1300:
                subtitle_text = "Professional Dermatology AI • Gemma 3n-E4B"

            subtitle_label = ctk.CTkLabel(
                header_frame,
                text=subtitle_text,
                font=ctk.CTkFont(size=subtitle_size),
                text_color="white"
            )
            subtitle_label.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        # Status indicators
        status_size = max(10, min(int(self.window_width / 120), 14))
        self.ai_status_label = ctk.CTkLabel(
            header_frame,
            text="🤖 AI: Initializing...",
            font=ctk.CTkFont(size=status_size),
            text_color="white"
        )
        self.ai_status_label.grid(row=0, column=2, padx=15, pady=10, sticky="e")
    
    def create_main_layout(self):
        """Create the main three-panel layout"""
        # Left panel - Image upload and controls
        self.create_left_panel()
        
        # Center panel - Image workspace
        self.create_center_panel()
        
        # Right panel - Analysis results
        self.create_right_panel()
    
    def create_left_panel(self):
        """Create responsive left control panel"""
        # Calculate responsive panel width
        panel_width = max(200, min(int(self.window_width * 0.25), 350))

        left_frame = ctk.CTkFrame(self.root, width=panel_width, fg_color=self.colors['surface'])
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_frame.grid_propagate(False)

        # Store reference for responsive updates
        self.left_frame = left_frame
        
        # Image Upload Section with responsive sizing
        label_size = max(12, min(int(self.window_width / 90), 18))
        upload_label = ctk.CTkLabel(
            left_frame,
            text="📷 IMAGE UPLOAD",
            font=ctk.CTkFont(size=label_size, weight="bold"),
            text_color=self.colors['text_primary']
        )
        upload_label.pack(pady=(15, 8))

        # Responsive button sizing
        button_height = max(35, min(int(self.window_height / 20), 50))
        button_font_size = max(11, min(int(self.window_width / 100), 16))

        # Responsive button text
        button_text = "📁 Load Image" if self.window_width < 1200 else "📁 Load Dermatological Image"

        self.upload_btn = ctk.CTkButton(
            left_frame,
            text=button_text,
            font=ctk.CTkFont(size=button_font_size, weight="bold"),
            fg_color=self.colors['primary'],
            hover_color=self.colors['secondary'],
            height=button_height,
            command=self.load_image
        )
        self.upload_btn.pack(pady=8, padx=15, fill="x")
        
        # Image info
        self.image_info_label = ctk.CTkLabel(
            left_frame,
            text="No image loaded",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text_secondary']
        )
        self.image_info_label.pack(pady=5)
        
        # Analysis Options Section
        options_label = ctk.CTkLabel(
            left_frame,
            text="⚙️ ANALYSIS OPTIONS",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['text_primary']
        )
        options_label.pack(pady=(30, 10))
        
        # Analysis mode selection
        self.analysis_mode = ctk.StringVar(value="multi")
        
        multi_radio = ctk.CTkRadioButton(
            left_frame,
            text="🔬 Multi-Condition Analysis (All 14+ conditions)",
            variable=self.analysis_mode,
            value="multi",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text_primary']
        )
        multi_radio.pack(pady=5, padx=20, anchor="w")
        
        single_radio = ctk.CTkRadioButton(
            left_frame,
            text="🎯 Single-Condition Analysis",
            variable=self.analysis_mode,
            value="single",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['text_primary'],
            command=self.on_analysis_mode_change
        )
        single_radio.pack(pady=5, padx=20, anchor="w")
        
        # Condition selection (for single mode)
        self.condition_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        self.condition_frame.pack(pady=10, padx=20, fill="x")
        
        self.condition_var = ctk.StringVar(value="Melanoma")
        self.condition_menu = ctk.CTkOptionMenu(
            self.condition_frame,
            variable=self.condition_var,
            values=[
                "Melanoma", "Basal Cell Carcinoma", "Squamous Cell Carcinoma",
                "Actinic Keratosis", "Melanocytic Nevus", "Seborrheic Keratosis",
                "Dermatofibroma", "Vascular Lesion", "Monkeypox", "Chickenpox",
                "Measles", "Hand, Foot, and Mouth Disease", "Cowpox", "Healthy Skin"
            ],
            font=ctk.CTkFont(size=11),
            state="disabled"
        )
        self.condition_menu.pack(fill="x")
        
        # Analysis Controls
        controls_label = ctk.CTkLabel(
            left_frame,
            text="🚀 ANALYSIS CONTROLS",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.colors['text_primary']
        )
        controls_label.pack(pady=(30, 10))
        
        self.analyze_btn = ctk.CTkButton(
            left_frame,
            text="🔬 Start AI Analysis",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors['success'],
            hover_color="#218838",
            height=50,
            state="disabled",
            command=self.start_analysis
        )
        self.analyze_btn.pack(pady=10, padx=20, fill="x")
        
        self.reset_btn = ctk.CTkButton(
            left_frame,
            text="🔄 Reset Analysis",
            font=ctk.CTkFont(size=12),
            fg_color=self.colors['warning'],
            hover_color="#e0a800",
            height=35,
            state="disabled",
            command=self.reset_analysis
        )
        self.reset_btn.pack(pady=5, padx=20, fill="x")
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(left_frame)
        self.progress_bar.pack(pady=10, padx=20, fill="x")
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(
            left_frame,
            text="Ready for analysis",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['text_secondary']
        )
        self.progress_label.pack(pady=5)
    
    def create_center_panel(self):
        """Create responsive center image workspace"""
        center_frame = ctk.CTkFrame(self.root, fg_color=self.colors['surface'])
        center_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=10)

        # Store reference for responsive updates
        self.center_frame = center_frame

        # Responsive workspace title
        title_size = max(12, min(int(self.window_width / 90), 20))
        workspace_text = "🖼️ WORKSPACE" if self.window_width < 1200 else "🖼️ IMAGE ANALYSIS WORKSPACE"

        workspace_label = ctk.CTkLabel(
            center_frame,
            text=workspace_text,
            font=ctk.CTkFont(size=title_size, weight="bold"),
            text_color=self.colors['text_primary']
        )
        workspace_label.pack(pady=(15, 8))

        # Responsive image display area
        padding = max(10, min(int(self.window_width / 70), 25))
        self.image_frame = ctk.CTkFrame(center_frame, fg_color=self.colors['background'])
        self.image_frame.pack(expand=True, fill="both", padx=padding, pady=padding)

        # Responsive placeholder text
        placeholder_size = max(12, min(int(self.window_width / 100), 18))
        placeholder_text = "📷 No image loaded\n\nClick 'Load Image' to begin" if self.window_width < 1200 else "📷 No image loaded\n\nClick 'Load Dermatological Image' to begin analysis"

        self.image_label = ctk.CTkLabel(
            self.image_frame,
            text=placeholder_text,
            font=ctk.CTkFont(size=placeholder_size),
            text_color=self.colors['text_secondary']
        )
        self.image_label.pack(expand=True)

        # Initialize responsive display dimensions
        self.display_width = max(300, min(int(self.window_width * 0.4), 600))
        self.display_height = max(250, min(int(self.window_height * 0.5), 500))
    
    def create_right_panel(self):
        """Create responsive right results panel"""
        # Calculate responsive panel width
        panel_width = max(250, min(int(self.window_width * 0.35), 450))

        right_frame = ctk.CTkFrame(self.root, width=panel_width, fg_color=self.colors['surface'])
        right_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 10), pady=10)
        right_frame.grid_propagate(False)

        # Store reference for responsive updates
        self.right_frame = right_frame

        # Responsive results title
        title_size = max(12, min(int(self.window_width / 90), 18))
        results_text = "📊 RESULTS" if self.window_width < 1200 else "📊 ANALYSIS RESULTS"

        results_label = ctk.CTkLabel(
            right_frame,
            text=results_text,
            font=ctk.CTkFont(size=title_size, weight="bold"),
            text_color=self.colors['text_primary']
        )
        results_label.pack(pady=(15, 8))

        # Responsive scrollable results area
        padding = max(10, min(int(self.window_width / 80), 25))
        self.results_scroll = ctk.CTkScrollableFrame(
            right_frame,
            fg_color=self.colors['background']
        )
        self.results_scroll.pack(expand=True, fill="both", padx=padding, pady=padding)

        # Responsive placeholder message
        placeholder_size = max(11, min(int(self.window_width / 110), 16))

        # Adjust placeholder text based on screen size
        if self.window_width < 1000:
            placeholder_text = "📋 Results will appear here\n\nLoad image and analyze to see:\n• Conditions\n• Recommendations\n• Confidence"
        else:
            placeholder_text = "📋 Analysis results will appear here\n\nLoad an image and start analysis to see:\n• Detected conditions\n• Risk assessment\n• Clinical recommendations\n• Confidence metrics"

        placeholder_label = ctk.CTkLabel(
            self.results_scroll,
            text=placeholder_text,
            font=ctk.CTkFont(size=placeholder_size),
            text_color=self.colors['text_secondary'],
            justify="center"
        )
        placeholder_label.pack(pady=30)
    
    def create_status_bar(self):
        """Create bottom status bar"""
        status_frame = ctk.CTkFrame(self.root, height=30, fg_color=self.colors['border'])
        status_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=0, pady=0)
        status_frame.grid_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🏥 DermAI Pro ready - Load an image to begin professional analysis",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['text_secondary']
        )
        self.status_label.pack(side="left", padx=10, pady=5)

    def initialize_ai_engine(self):
        """Initialize AI engine in background"""
        def init_worker():
            try:
                self.update_status("Initializing AI engine...")
                success = self.ai_engine.initialize_model()

                if success:
                    self.root.after(0, lambda: self.ai_status_label.configure(text="🤖 AI: Ready"))
                    self.update_status("AI engine ready - Load an image to begin analysis")
                else:
                    self.root.after(0, lambda: self.ai_status_label.configure(text="🤖 AI: Error"))
                    self.update_status("AI engine initialization failed")

                    # Show detailed error message
                    self.root.after(0, lambda: messagebox.showwarning(
                        "AI Engine Error",
                        "AI engine initialization failed!\n\n"
                        "Common solutions:\n"
                        "1. Ensure Ollama is running: ollama serve\n"
                        "2. Install Gemma model: ollama pull gemma3n:e4b\n"
                        "3. Wait for model to load (can take 1-2 minutes)\n"
                        "4. Check if port 11434 is available\n\n"
                        "You can still load images, but analysis will not work."
                    ))

            except Exception as e:
                logger.error(f"AI initialization failed: {e}")
                self.root.after(0, lambda: self.ai_status_label.configure(text="🤖 AI: Error"))
                self.update_status(f"AI initialization error: {str(e)}")

        # Start initialization in background
        init_thread = threading.Thread(target=init_worker, daemon=True)
        init_thread.start()

    def on_analysis_mode_change(self):
        """Handle analysis mode change"""
        if self.analysis_mode.get() == "single":
            self.condition_menu.configure(state="normal")
        else:
            self.condition_menu.configure(state="disabled")

    def load_image(self):
        """Load dermatological image for analysis"""
        try:
            file_types = [
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]

            file_path = filedialog.askopenfilename(
                title="Select Dermatological Image",
                filetypes=file_types,
                initialdir=str(Path.home() / "Pictures")
            )

            if not file_path:
                return

            self.update_status("📷 Loading image...")

            # Load and validate image
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Error", "Could not load image. Please select a valid image file.")
                return

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Store image data
            self.current_image = image_rgb
            self.current_image_path = file_path

            # Display image in workspace
            self.display_image(image_rgb)

            # Update UI state
            self.analyze_btn.configure(state="normal")
            self.reset_btn.configure(state="normal")

            # Update image info
            height, width = image_rgb.shape[:2]
            file_size = Path(file_path).stat().st_size / 1024  # KB
            self.image_info_label.configure(
                text=f"{width}x{height} pixels • {file_size:.1f} KB"
            )

            self.update_status(f"✅ Image loaded: {Path(file_path).name}")
            logger.info(f"📷 Image loaded: {file_path} ({width}x{height})")

        except Exception as e:
            logger.error(f"❌ Image loading failed: {e}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.update_status("❌ Image loading failed")

    def display_image(self, image_rgb: np.ndarray):
        """Display image in the workspace with responsive sizing"""
        try:
            # Use responsive display dimensions
            display_width = self.display_width
            display_height = self.display_height

            height, width = image_rgb.shape[:2]
            aspect_ratio = width / height

            # Calculate new dimensions maintaining aspect ratio
            if aspect_ratio > display_width / display_height:
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
            else:
                new_height = display_height
                new_width = int(display_height * aspect_ratio)

            # Ensure minimum size for very small screens
            new_width = max(200, new_width)
            new_height = max(150, new_height)

            # Resize image for display
            display_image = cv2.resize(image_rgb, (new_width, new_height))

            # Convert to PIL Image
            pil_image = Image.fromarray(display_image)

            # Use CTkImage with responsive sizing
            ctk_image = ctk.CTkImage(
                light_image=pil_image,
                dark_image=pil_image,
                size=(new_width, new_height)
            )

            # Update image label
            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image  # Keep a reference

            # Store current display dimensions for future reference
            self.current_display_width = new_width
            self.current_display_height = new_height

        except Exception as e:
            logger.error(f"Image display failed: {e}")
            self.image_label.configure(text="Image display error")

    def start_analysis(self):
        """Start AI analysis of the loaded image"""
        try:
            if self.current_image is None:
                messagebox.showwarning("Warning", "Please load an image first.")
                return

            if not self.ai_engine.model_ready:
                messagebox.showerror("Error", "AI engine not ready. Please wait for initialization.")
                return

            if self.processing:
                messagebox.showinfo("Info", "Analysis already in progress.")
                return

            # Update UI state
            self.processing = True
            self.analyze_btn.configure(state="disabled", text="🔄 Analyzing...")
            self.reset_btn.configure(state="disabled")
            self.progress_bar.set(0)

            # Clear previous results
            self.clear_results()

            # Start analysis in background thread
            analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
            analysis_thread.start()

        except Exception as e:
            logger.error(f"❌ Analysis start failed: {e}")
            messagebox.showerror("Error", f"Failed to start analysis: {str(e)}")
            self.processing = False
            self.analyze_btn.configure(state="normal", text="🔬 Start AI Analysis")
            self.reset_btn.configure(state="normal")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        try:
            logger.info("🔬 Starting comprehensive dermatological analysis...")

            # Step 1: Lesion Detection
            self.update_progress("🔍 Detecting skin lesions...", 0.2)
            detection_results = self.lesion_detector.detect_lesions(
                self.current_image,
                detection_mode="comprehensive"
            )

            if not detection_results.get('success', False):
                self.analysis_error("Lesion detection failed")
                return

            lesions = detection_results.get('lesions', [])
            logger.info(f"✅ Detected {len(lesions)} potential lesions")

            # Step 2: AI Analysis
            analysis_type = self.analysis_mode.get()
            target_condition = self.condition_var.get() if analysis_type == "single" else None

            self.update_progress("🤖 Running AI analysis...", 0.6)

            ai_results = self.ai_engine.analyze_image(
                self.current_image,
                analysis_type=analysis_type,
                target_condition=target_condition
            )

            if not ai_results.get('success', False):
                self.analysis_error(f"AI analysis failed: {ai_results.get('error', 'Unknown error')}")
                return

            # Step 3: Generate Clinical Recommendations
            self.update_progress("📋 Generating recommendations...", 0.9)

            clinical_recommendations = self.ai_engine.get_clinical_recommendations(ai_results)

            # Combine all results
            self.analysis_results = {
                'success': True,
                'timestamp': time.time(),
                'detection_results': detection_results,
                'ai_analysis': ai_results,
                'clinical_recommendations': clinical_recommendations,
                'analysis_type': analysis_type,
                'target_condition': target_condition
            }

            # Update UI with results
            self.update_progress("✅ Analysis complete!", 1.0)
            self.root.after(100, self.display_results)

            logger.info("✅ Comprehensive analysis completed successfully")

        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            self.analysis_error(f"Analysis error: {str(e)}")

        finally:
            # Reset UI state
            self.processing = False
            self.root.after(0, lambda: self.analyze_btn.configure(
                state="normal", text="🔬 Start AI Analysis"
            ))
            self.root.after(0, lambda: self.reset_btn.configure(state="normal"))

    def analysis_error(self, error_message: str):
        """Handle analysis error"""
        self.root.after(0, lambda: self.update_status(f"❌ {error_message}"))
        self.root.after(0, lambda: messagebox.showerror("Analysis Error", error_message))
        self.processing = False

    def display_results(self):
        """Display comprehensive analysis results"""
        try:
            # Clear previous results
            for widget in self.results_scroll.winfo_children():
                widget.destroy()

            if not self.analysis_results or not self.analysis_results.get('success', False):
                error_label = ctk.CTkLabel(
                    self.results_scroll,
                    text="❌ No analysis results available",
                    font=ctk.CTkFont(size=14),
                    text_color=self.colors['danger']
                )
                error_label.pack(pady=20)
                return

            # Results header
            header_label = ctk.CTkLabel(
                self.results_scroll,
                text="🎯 AI Analysis Results",
                font=ctk.CTkFont(size=18, weight="bold"),
                text_color=self.colors['text_primary']
            )
            header_label.pack(pady=(10, 20))

            # Display results based on analysis type
            ai_analysis = self.analysis_results.get('ai_analysis', {})
            analysis_type = ai_analysis.get('analysis_type', 'multi')

            if analysis_type == 'multi':
                self.display_multi_condition_results(ai_analysis)
            else:
                self.display_single_condition_results(ai_analysis)

            # Display clinical recommendations
            self.display_clinical_recommendations()

            # Display confidence metrics
            self.display_confidence_metrics(ai_analysis)

            # Display medical disclaimer
            self.display_medical_disclaimer()

        except Exception as e:
            logger.error(f"❌ Results display failed: {e}")
            error_label = ctk.CTkLabel(
                self.results_scroll,
                text=f"❌ Results display error: {str(e)}",
                font=ctk.CTkFont(size=12),
                text_color=self.colors['danger']
            )
            error_label.pack(pady=20)

    def display_multi_condition_results(self, ai_analysis: Dict[str, Any]):
        """Display multi-condition analysis results with responsive design"""
        try:
            # Responsive conditions detected section
            padding = max(5, min(int(self.window_width / 140), 15))
            conditions_frame = ctk.CTkFrame(self.results_scroll, fg_color=self.colors['background'])
            conditions_frame.pack(fill="x", pady=padding, padx=padding)

            # Responsive title
            title_size = max(11, min(int(self.window_width / 100), 16))
            title_text = "🔬 CONDITIONS" if self.window_width < 1200 else "🔬 CONDITIONS DETECTED"

            conditions_title = ctk.CTkLabel(
                conditions_frame,
                text=title_text,
                font=ctk.CTkFont(size=title_size, weight="bold"),
                text_color=self.colors['text_primary']
            )
            conditions_title.pack(anchor="w", padx=padding, pady=(padding, 5))

            conditions_detected = ai_analysis.get('conditions_detected', {})

            if conditions_detected:
                # Sort by probability
                sorted_conditions = sorted(
                    conditions_detected.items(),
                    key=lambda x: x[1].get('probability', 0),
                    reverse=True
                )

                for condition_id, condition_data in sorted_conditions[:8]:  # Show top 8
                    probability = condition_data.get('probability', 0.0)
                    confidence = condition_data.get('confidence', 0.0)
                    condition_name = condition_data.get('name', condition_id.replace('_', ' ').title())
                    risk_level = condition_data.get('risk_level', 'low')

                    if probability > 0.05:  # Show conditions with >5% probability
                        # Color based on risk level
                        if risk_level == 'high':
                            text_color = self.colors['danger']
                        elif risk_level == 'medium':
                            text_color = self.colors['warning']
                        else:
                            text_color = self.colors['text_secondary']

                        condition_text = f"• {condition_name}: {probability*100:.1f}% (confidence: {confidence*100:.1f}%)"

                        condition_label = ctk.CTkLabel(
                            conditions_frame,
                            text=condition_text,
                            font=ctk.CTkFont(size=12),
                            text_color=text_color
                        )
                        condition_label.pack(anchor="w", padx=25, pady=2)
            else:
                no_conditions_label = ctk.CTkLabel(
                    conditions_frame,
                    text="• No specific conditions detected with high confidence",
                    font=ctk.CTkFont(size=12),
                    text_color=self.colors['text_secondary']
                )
                no_conditions_label.pack(anchor="w", padx=25, pady=5)

        except Exception as e:
            logger.error(f"❌ Multi-condition display failed: {e}")

    def display_single_condition_results(self, ai_analysis: Dict[str, Any]):
        """Display single condition analysis results"""
        try:
            single_analysis = ai_analysis.get('single_condition_analysis', {})

            # Single condition section
            condition_frame = ctk.CTkFrame(self.results_scroll, fg_color=self.colors['background'])
            condition_frame.pack(fill="x", pady=10, padx=10)

            condition_name = single_analysis.get('condition_name', 'Unknown')
            probability = single_analysis.get('probability', 0.0)
            confidence = single_analysis.get('confidence', 0.0)
            risk_level = single_analysis.get('risk_level', 'low')

            # Title
            condition_title = ctk.CTkLabel(
                condition_frame,
                text=f"🎯 {condition_name.upper()} ANALYSIS",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.colors['text_primary']
            )
            condition_title.pack(anchor="w", padx=15, pady=(10, 5))

            # Probability and confidence
            prob_text = f"• Probability: {probability*100:.1f}%"
            conf_text = f"• Confidence: {confidence*100:.1f}%"
            risk_text = f"• Risk Level: {risk_level.upper()}"

            # Color based on risk
            if risk_level == 'high':
                risk_color = self.colors['danger']
            elif risk_level == 'medium':
                risk_color = self.colors['warning']
            else:
                risk_color = self.colors['success']

            for text, color in [(prob_text, self.colors['text_secondary']),
                               (conf_text, self.colors['text_secondary']),
                               (risk_text, risk_color)]:
                label = ctk.CTkLabel(
                    condition_frame,
                    text=text,
                    font=ctk.CTkFont(size=12),
                    text_color=color
                )
                label.pack(anchor="w", padx=25, pady=2)

            # Clinical assessment
            clinical_assessment = single_analysis.get('clinical_assessment', '')
            if clinical_assessment:
                assessment_label = ctk.CTkLabel(
                    condition_frame,
                    text=f"• Clinical Assessment: {clinical_assessment}",
                    font=ctk.CTkFont(size=12),
                    text_color=self.colors['text_secondary'],
                    wraplength=350
                )
                assessment_label.pack(anchor="w", padx=25, pady=5)

        except Exception as e:
            logger.error(f"❌ Single condition display failed: {e}")

    def display_clinical_recommendations(self):
        """Display clinical recommendations"""
        try:
            recommendations = self.analysis_results.get('clinical_recommendations', [])

            if recommendations:
                rec_frame = ctk.CTkFrame(self.results_scroll, fg_color=self.colors['background'])
                rec_frame.pack(fill="x", pady=10, padx=10)

                rec_title = ctk.CTkLabel(
                    rec_frame,
                    text="📋 CLINICAL RECOMMENDATIONS",
                    font=ctk.CTkFont(size=14, weight="bold"),
                    text_color=self.colors['text_primary']
                )
                rec_title.pack(anchor="w", padx=15, pady=(10, 5))

                for i, recommendation in enumerate(recommendations[:6], 1):
                    rec_text = f"{i}. {recommendation}"
                    rec_label = ctk.CTkLabel(
                        rec_frame,
                        text=rec_text,
                        font=ctk.CTkFont(size=12),
                        text_color=self.colors['text_secondary'],
                        wraplength=350
                    )
                    rec_label.pack(anchor="w", padx=25, pady=2)

        except Exception as e:
            logger.error(f"❌ Recommendations display failed: {e}")

    def display_confidence_metrics(self, ai_analysis: Dict[str, Any]):
        """Display analysis confidence metrics"""
        try:
            metrics_frame = ctk.CTkFrame(self.results_scroll, fg_color=self.colors['background'])
            metrics_frame.pack(fill="x", pady=10, padx=10)

            metrics_title = ctk.CTkLabel(
                metrics_frame,
                text="📊 ANALYSIS CONFIDENCE",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.colors['text_primary']
            )
            metrics_title.pack(anchor="w", padx=15, pady=(10, 5))

            processing_time = ai_analysis.get('processing_time', 0)
            model_used = ai_analysis.get('model_used', 'Unknown')

            metrics = [
                f"• Processing Time: {processing_time:.1f} seconds",
                f"• AI Model: {model_used}",
                f"• Analysis Method: Professional AI Assessment"
            ]

            for metric in metrics:
                metric_label = ctk.CTkLabel(
                    metrics_frame,
                    text=metric,
                    font=ctk.CTkFont(size=12),
                    text_color=self.colors['text_secondary']
                )
                metric_label.pack(anchor="w", padx=25, pady=2)

        except Exception as e:
            logger.error(f"❌ Confidence metrics display failed: {e}")

    def display_medical_disclaimer(self):
        """Display important medical disclaimer"""
        disclaimer_frame = ctk.CTkFrame(self.results_scroll, fg_color=self.colors['warning'])
        disclaimer_frame.pack(fill="x", pady=15, padx=10)

        disclaimer_text = (
            "⚠️ MEDICAL DISCLAIMER\n\n"
            "This AI analysis is for professional medical assistance only. "
            "Results should always be validated by qualified dermatologists. "
            "Not intended for self-diagnosis or replacing professional medical advice. "
            "Consult a healthcare provider for definitive diagnosis and treatment."
        )

        disclaimer_label = ctk.CTkLabel(
            disclaimer_frame,
            text=disclaimer_text,
            font=ctk.CTkFont(size=11),
            text_color="white",
            wraplength=350,
            justify="left"
        )
        disclaimer_label.pack(padx=15, pady=10)

    def clear_results(self):
        """Clear previous analysis results"""
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        placeholder_label = ctk.CTkLabel(
            self.results_scroll,
            text="🔄 Analysis in progress...\n\nPlease wait while the AI model\nprocesses your image.",
            font=ctk.CTkFont(size=14),
            text_color=self.colors['text_secondary'],
            justify="center"
        )
        placeholder_label.pack(pady=50)

    def reset_analysis(self):
        """Reset the analysis interface"""
        try:
            # Clear current data
            self.current_image = None
            self.current_image_path = None
            self.analysis_results = None

            # Reset image display
            self.image_label.configure(
                image=None,
                text="📷 No image loaded\n\nClick 'Load Dermatological Image' to begin analysis"
            )

            # Reset image info
            self.image_info_label.configure(text="No image loaded")

            # Clear results
            for widget in self.results_scroll.winfo_children():
                widget.destroy()

            placeholder_label = ctk.CTkLabel(
                self.results_scroll,
                text="📋 Analysis results will appear here\n\nLoad an image and start analysis to see:\n• Detected conditions\n• Risk assessment\n• Clinical recommendations\n• Confidence metrics",
                font=ctk.CTkFont(size=14),
                text_color=self.colors['text_secondary'],
                justify="center"
            )
            placeholder_label.pack(pady=50)

            # Reset progress
            self.progress_bar.set(0)
            self.progress_label.configure(text="Ready for analysis")

            # Reset button states
            self.analyze_btn.configure(state="disabled", text="🔬 Start AI Analysis")
            self.reset_btn.configure(state="disabled")

            self.update_status("🔄 Interface reset - ready for new analysis")
            logger.info("🔄 Analysis interface reset")

        except Exception as e:
            logger.error(f"❌ Reset failed: {e}")
            messagebox.showerror("Error", f"Reset failed: {str(e)}")

    def update_progress(self, text: str, progress: float):
        """Update progress bar and text"""
        self.root.after(0, lambda: self.progress_bar.set(progress))
        self.root.after(0, lambda: self.progress_label.configure(text=text))
        self.root.after(0, lambda: self.update_status(text))

    def update_status(self, message: str):
        """Update status bar message"""
        if self.root and self.status_label:
            self.root.after(0, lambda: self.status_label.configure(text=f"🏥 {message}"))
