/**
 * Admin Controls for Spiritual Q&A System
 * Handles the advanced parameter tweaking functionality
 */

class AdminControls {
    constructor(app) {
        this.app = app;
        this.diversitySlider = document.getElementById('diversity-slider');
        this.diversityValue = document.getElementById('diversity-value');
        this.kSlider = document.getElementById('k-slider');
        this.kValue = document.getElementById('k-value');
        this.useMMRToggle = document.getElementById('use-mmr-toggle');
        this.mmrStatus = document.getElementById('mmr-status');
        this.saveButton = document.getElementById('save-admin-settings');
        this.presetButtons = document.querySelectorAll('.preset-button');
        
        // Default values
        this.adminSettings = {
            diversity: 0.6,
            k: 5,
            useMMR: true
        };
        
        // Try to load saved settings
        this.loadSettings();
        this.initEventListeners();
    }
    
    initEventListeners() {
        // Diversity slider
        if (this.diversitySlider) {
            this.diversitySlider.addEventListener('input', () => {
                const value = parseFloat(this.diversitySlider.value) / 100;
                this.updateDiversityValue(value);
            });
        }
        
        // k slider
        if (this.kSlider) {
            this.kSlider.addEventListener('input', () => {
                const value = parseInt(this.kSlider.value);
                this.updateKValue(value);
            });
        }
        
        // MMR toggle
        if (this.useMMRToggle) {
            this.useMMRToggle.addEventListener('change', () => {
                const isChecked = this.useMMRToggle.checked;
                this.updateMMRStatus(isChecked);
            });
        }
        
        // Save button
        if (this.saveButton) {
            this.saveButton.addEventListener('click', () => {
                this.saveSettings();
                this.app.showNotification('Settings saved successfully!', 'success');
            });
        }
        
        // Preset buttons
        if (this.presetButtons) {
            this.presetButtons.forEach(button => {
                button.addEventListener('click', (e) => {
                    const preset = e.target.dataset.preset;
                    this.applyPreset(preset);
                });
            });
        }
    }
    
    updateDiversityValue(value) {
        this.adminSettings.diversity = value;
        if (this.diversityValue) {
            this.diversityValue.textContent = value.toFixed(1);
        }
    }
    
    updateKValue(value) {
        this.adminSettings.k = value;
        if (this.kValue) {
            this.kValue.textContent = value;
        }
    }
    
    updateMMRStatus(isChecked) {
        this.adminSettings.useMMR = isChecked;
        if (this.mmrStatus) {
            this.mmrStatus.textContent = isChecked ? 'Enabled' : 'Disabled';
        }
        
        // Disable/enable diversity slider based on MMR status
        if (this.diversitySlider) {
            this.diversitySlider.disabled = !isChecked;
            this.diversitySlider.parentNode.classList.toggle('disabled', !isChecked);
        }
    }
    
    applyPreset(preset) {
        // Get preset values from the app's retrieval params function
        const params = this.app.getRetrievalParamsFromStyle(preset);
        
        // Update UI
        const diversityVal = parseFloat(params.diversity);
        if (this.diversitySlider) {
            this.diversitySlider.value = diversityVal * 100;
            this.updateDiversityValue(diversityVal);
        }
        
        if (this.kSlider) {
            this.kSlider.value = params.k;
            this.updateKValue(params.k);
        }
        
        if (this.useMMRToggle) {
            this.useMMRToggle.checked = params.use_mmr;
            this.updateMMRStatus(params.use_mmr);
        }
        
        // Highlight active preset
        this.presetButtons.forEach(button => {
            button.classList.toggle('active', button.dataset.preset === preset);
        });
        
        this.app.showNotification(`Applied ${preset} reading style preset`, 'info');
    }
    
    saveSettings() {
        // Store settings in local storage
        localStorage.setItem('admin_settings', JSON.stringify(this.adminSettings));
    }
    
    loadSettings() {
        // Load settings from local storage
        try {
            const stored = localStorage.getItem('admin_settings');
            if (stored) {
                const settings = JSON.parse(stored);
                this.adminSettings = { ...this.adminSettings, ...settings };
                
                // Apply loaded settings to UI
                this.updateUIFromSettings();
            }
        } catch (error) {
            console.error('Error loading admin settings:', error);
        }
    }
    
    updateUIFromSettings() {
        if (this.diversitySlider) {
            this.diversitySlider.value = this.adminSettings.diversity * 100;
            this.updateDiversityValue(this.adminSettings.diversity);
        }
        
        if (this.kSlider) {
            this.kSlider.value = this.adminSettings.k;
            this.updateKValue(this.adminSettings.k);
        }
        
        if (this.useMMRToggle) {
            this.useMMRToggle.checked = this.adminSettings.useMMR;
            this.updateMMRStatus(this.adminSettings.useMMR);
        }
    }
    
    // Get current admin settings for API calls
    getCurrentSettings() {
        return {
            use_mmr: this.adminSettings.useMMR,
            diversity: this.adminSettings.diversity,
            k: this.adminSettings.k
        };
    }
}

// Export for use in main app
window.AdminControls = AdminControls;
