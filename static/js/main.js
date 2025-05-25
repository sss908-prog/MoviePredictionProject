/**
 * Main JavaScript file for Movie Rating Predictor
 * Handles interactions, animations, and utility functions
 */

// Global application object
const MoviePredictor = {
    // Configuration
    config: {
        animationDuration: 300,
        chartColors: [
            'rgba(54, 162, 235, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(255, 99, 132, 0.8)',
            'rgba(255, 206, 86, 0.8)',
            'rgba(153, 102, 255, 0.8)'
        ]
    },
    
    // Initialize the application
    init: function() {
        this.setupEventListeners();
        this.initializeAnimations();
        this.setupFormValidation();
        this.initializeTooltips();
        this.setupScrollEffects();
        
        console.log('Movie Predictor initialized successfully');
    },
    
    // Set up event listeners
    setupEventListeners: function() {
        // Form submission handlers
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', this.handleFormSubmission.bind(this));
        });
        
        // Navigation active state
        this.updateNavigationState();
        
        // Card hover effects
        this.setupCardHoverEffects();
        
        // Button click effects
        this.setupButtonEffects();
    },
    
    // Handle form submissions with loading states
    handleFormSubmission: function(event) {
        const form = event.target;
        const submitButton = form.querySelector('button[type="submit"]');
        
        if (submitButton) {
            const originalText = submitButton.innerHTML;
            const loadingText = submitButton.dataset.loading || 
                '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
            
            // Show loading state
            submitButton.innerHTML = loadingText;
            submitButton.disabled = true;
            
            // Add loading class to form
            form.classList.add('loading');
            
            // If validation fails, restore button state
            setTimeout(() => {
                if (!form.checkValidity()) {
                    submitButton.innerHTML = originalText;
                    submitButton.disabled = false;
                    form.classList.remove('loading');
                }
            }, 100);
        }
    },
    
    // Initialize animations
    initializeAnimations: function() {
        // Fade in elements on scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in-up');
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);
        
        // Observe cards and important elements
        document.querySelectorAll('.card, .hero-section, .table').forEach(el => {
            observer.observe(el);
        });
    },
    
    // Set up form validation
    setupFormValidation: function() {
        const forms = document.querySelectorAll('.needs-validation');
        
        forms.forEach(form => {
            form.addEventListener('submit', (event) => {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            });
        });
        
        // Real-time validation for numeric inputs
        const numericInputs = document.querySelectorAll('input[type="number"]');
        numericInputs.forEach(input => {
            input.addEventListener('input', this.validateNumericInput.bind(this));
            input.addEventListener('blur', this.validateNumericInput.bind(this));
        });
    },
    
    // Validate numeric input
    validateNumericInput: function(event) {
        const input = event.target;
        const value = parseFloat(input.value);
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        let isValid = true;
        let message = '';
        
        if (isNaN(value)) {
            isValid = false;
            message = 'Please enter a valid number';
        } else if (value < min) {
            isValid = false;
            message = `Value must be at least ${min}`;
        } else if (value > max) {
            isValid = false;
            message = `Value must be no more than ${max}`;
        }
        
        // Update validation state
        if (isValid) {
            input.setCustomValidity('');
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        } else {
            input.setCustomValidity(message);
            input.classList.remove('is-valid');
            input.classList.add('is-invalid');
        }
        
        // Update feedback message
        let feedback = input.parentNode.querySelector('.invalid-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            input.parentNode.appendChild(feedback);
        }
        feedback.textContent = message;
    },
    
    // Initialize tooltips
    initializeTooltips: function() {
        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(
                document.querySelectorAll('[data-bs-toggle="tooltip"]')
            );
            tooltipTriggerList.map(function(tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    },
    
    // Set up scroll effects
    setupScrollEffects: function() {
        let lastScrollTop = 0;
        const navbar = document.querySelector('.navbar');
        
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            // Navbar hide/show on scroll
            if (navbar) {
                if (scrollTop > lastScrollTop && scrollTop > 100) {
                    navbar.style.transform = 'translateY(-100%)';
                } else {
                    navbar.style.transform = 'translateY(0)';
                }
            }
            
            lastScrollTop = scrollTop;
        });
    },
    
    // Update navigation active state
    updateNavigationState: function() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    },
    
    // Set up card hover effects
    setupCardHoverEffects: function() {
        const cards = document.querySelectorAll('.card');
        
        cards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-4px)';
                this.style.transition = 'transform 0.3s ease';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0)';
            });
        });
    },
    
    // Set up button effects
    setupButtonEffects: function() {
        const buttons = document.querySelectorAll('.btn');
        
        buttons.forEach(button => {
            button.addEventListener('click', function(e) {
                // Create ripple effect
                const ripple = document.createElement('span');
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    width: ${size}px;
                    height: ${size}px;
                    left: ${x}px;
                    top: ${y}px;
                    background: rgba(255, 255, 255, 0.4);
                    border-radius: 50%;
                    transform: scale(0);
                    animation: ripple 0.6s ease-out;
                    pointer-events: none;
                `;
                
                this.style.position = 'relative';
                this.style.overflow = 'hidden';
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    },
    
    // Utility functions
    utils: {
        // Format numbers with proper decimal places
        formatNumber: function(num, decimals = 2) {
            return parseFloat(num).toFixed(decimals);
        },
        
        // Format currency
        formatCurrency: function(amount) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(amount * 1000000); // Convert millions to actual amount
        },
        
        // Format runtime
        formatRuntime: function(minutes) {
            const hours = Math.floor(minutes / 60);
            const mins = minutes % 60;
            return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
        },
        
        // Get rating category
        getRatingCategory: function(rating) {
            if (rating >= 8.0) return { category: 'Excellent', color: 'success' };
            if (rating >= 7.0) return { category: 'Good', color: 'primary' };
            if (rating >= 6.0) return { category: 'Average', color: 'warning' };
            if (rating >= 4.0) return { category: 'Below Average', color: 'secondary' };
            return { category: 'Poor', color: 'danger' };
        },
        
        // Show notification
        showNotification: function(message, type = 'info', duration = 5000) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
            alertDiv.style.cssText = `
                top: 100px;
                right: 20px;
                z-index: 1050;
                min-width: 300px;
            `;
            
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.appendChild(alertDiv);
            
            // Auto-remove after duration
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, duration);
        },
        
        // Smooth scroll to element
        scrollToElement: function(element, offset = 80) {
            const elementTop = element.offsetTop - offset;
            window.scrollTo({
                top: elementTop,
                behavior: 'smooth'
            });
        },
        
        // Debounce function
        debounce: function(func, wait, immediate) {
            let timeout;
            return function executedFunction() {
                const context = this;
                const args = arguments;
                const later = function() {
                    timeout = null;
                    if (!immediate) func.apply(context, args);
                };
                const callNow = immediate && !timeout;
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
                if (callNow) func.apply(context, args);
            };
        }
    }
};

// CSS for ripple animation
const rippleCSS = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;

// Add CSS to document
const style = document.createElement('style');
style.textContent = rippleCSS;
document.head.appendChild(style);

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    MoviePredictor.init();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden');
    } else {
        console.log('Page visible');
        // Refresh any time-sensitive data if needed
    }
});

// Handle window resize
window.addEventListener('resize', MoviePredictor.utils.debounce(function() {
    // Handle responsive adjustments
    console.log('Window resized');
}, 250));

// Export for use in other scripts
window.MoviePredictor = MoviePredictor;
