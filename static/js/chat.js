// State
let conversationId = null;
let currentImage = null;
let isLoading = false;
let currentTurn = 0;
let uxConfig = null;
let relevanceThreshold = 0.0; // Default: show all components
let currentSourceImages = null; // Store for threshold updates

// Carousel State
let carouselState = {
    images: [],           // All images from source
    currentIndex: 0,
    totalImages: 0
};

// API Base URL
const API_BASE = window.location.origin;

// DOM Elements
const messagesContainer = document.getElementById('messages');
const emptyState = document.getElementById('emptyState');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const sourcesList = document.getElementById('sourcesList');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const previewName = document.getElementById('previewName');

// Modal Elements
const imageModal = document.getElementById('imageModal');
const modalTitle = document.getElementById('modalTitle');
const modalImage = document.getElementById('modalImage');
const modalContainer = document.getElementById('modalContainer');
const modalLegend = document.getElementById('modalLegend');
const carouselStrip = document.getElementById('carouselStrip');

// Load UX config on page load
async function loadUXConfig() {
    try {
        const response = await fetch(`${API_BASE}/config/ux`);
        if (response.ok) {
            uxConfig = await response.json();
            console.log('UX Config loaded:', uxConfig);
        } else {
            console.warn('Failed to load UX config, using defaults');
        }
    } catch (error) {
        console.error('Error loading UX config:', error);
    }
}

// Determine visual tier based on relevance score
function getVisualTier(relevanceScore) {
    if (!uxConfig) {
        // Fallback default styling if config not loaded
        return {
            opacity: 0.7,
            color: '#D4A574',
            border_width: 2,
            font_size: '11px',
            font_weight: '400'
        };
    }

    const tiers = uxConfig.component_relevance_tiers;

    if (relevanceScore >= tiers.highly_relevant) {
        return uxConfig.component_visual_tiers.highly_relevant;
    } else if (relevanceScore >= tiers.relevant) {
        return uxConfig.component_visual_tiers.relevant;
    } else if (relevanceScore >= tiers.somewhat_relevant) {
        return uxConfig.component_visual_tiers.somewhat_relevant;
    } else {
        return uxConfig.component_visual_tiers.low_relevant;
    }
}

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
});

// Send on Enter (Shift+Enter for newline)
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Suggestion buttons (dynamically added)
function attachSuggestionListeners() {
    document.querySelectorAll('.suggestion').forEach(btn => {
        btn.addEventListener('click', function() {
            messageInput.value = this.dataset.query;
            sendMessage();
        });
    });
}

// Populate empty state with capabilities and suggestions
function populateEmptyState() {
    const defaultCapabilities = [
        { icon: '📄', label: 'Document Search' },
        { icon: '🔍', label: 'Visual Search' },
        { icon: '⚡', label: 'Fast Answers' },
        { icon: '🎯', label: 'Precise Results' }
    ];

    const defaultSuggestions = [
        'How do I troubleshoot errors?',
        'Show me installation instructions',
        'Find maintenance procedures',
        'Search for safety guidelines'
    ];

    // Populate capability pills
    const pillsContainer = document.getElementById('capabilityPills');
    if (pillsContainer) {
        pillsContainer.innerHTML = defaultCapabilities
            .map(cap => `
                <div class="capability-pill">
                    <span>${cap.icon}</span>
                    <span>${cap.label}</span>
                </div>
            `)
            .join('');
    }

    // Populate suggestions
    const suggestionsContainer = document.querySelector('.suggestions');
    if (suggestionsContainer) {
        suggestionsContainer.innerHTML = defaultSuggestions
            .map(suggestion => `
                <button class="suggestion" data-query="${suggestion}">
                    ${suggestion}
                </button>
            `)
            .join('');

        // Re-attach listeners to newly created suggestion buttons
        attachSuggestionListeners();
    }
}

// Initial population on page load
populateEmptyState();

// Image upload handling
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            currentImage = file;
            previewImage.src = e.target.result;
            previewName.textContent = file.name;
            imagePreview.style.display = 'flex';
        };
        reader.readAsDataURL(file);
    }
}

function clearImagePreview() {
    currentImage = null;
    imagePreview.style.display = 'none';
    document.getElementById('fileInput').value = '';
}

// Toggle sources panel (mobile)
function toggleSourcesPanel() {
    const panel = document.getElementById('sourcesPanel');
    panel.classList.toggle('mobile-visible');
}

// Modal Functions - Enhanced with Carousel
function openImageModal(source, startIndex = 0) {
    // If source is a single object, treat as single-image carousel
    // Otherwise, expect array of images from same document
    if (!Array.isArray(source)) {
        // Single image - check if there are other images from same document
        // For now, wrap in array
        carouselState.images = [source];
        carouselState.currentIndex = 0;
    } else {
        carouselState.images = source;
        carouselState.currentIndex = startIndex;
    }
    carouselState.totalImages = carouselState.images.length;

    renderCarouselUI();
    updateCarouselPosition();

    imageModal.classList.add('active');
    imageModal.style.display = 'flex';
}

function renderCarouselUI() {
    const currentSource = carouselState.images[carouselState.currentIndex];

    modalTitle.textContent = currentSource.title || 'Image View';
    modalImage.src = currentSource.url;

    // Clear previous overlays (keep the image)
    const existingOverlays = modalContainer.querySelectorAll('.bbox-overlay');
    existingOverlays.forEach(el => el.remove());

    // Clear previous navigation elements
    const existingNav = modalContainer.querySelectorAll('.carousel-nav, .carousel-counter');
    existingNav.forEach(el => el.remove());

    // Add navigation arrows if multiple images
    if (carouselState.totalImages > 1) {
        const prevBtn = document.createElement('button');
        prevBtn.className = 'carousel-nav prev';
        prevBtn.setAttribute('aria-label', 'Previous image (← or click)');
        prevBtn.setAttribute('title', 'Previous image (←)');
        prevBtn.onclick = (e) => { e.stopPropagation(); navigateCarousel(-1); };

        const nextBtn = document.createElement('button');
        nextBtn.className = 'carousel-nav next';
        nextBtn.setAttribute('aria-label', 'Next image (→ or click)');
        nextBtn.setAttribute('title', 'Next image (→)');
        nextBtn.onclick = (e) => { e.stopPropagation(); navigateCarousel(1); };

        modalContainer.appendChild(prevBtn);
        modalContainer.appendChild(nextBtn);

        // Add page counter
        const counter = document.createElement('div');
        counter.className = 'carousel-counter';
        counter.textContent = `${carouselState.currentIndex + 1} / ${carouselState.totalImages}`;
        modalContainer.appendChild(counter);

        // Render thumbnail strip
        renderThumbnailStrip();
    } else {
        // Hide thumbnail strip for single image
        if (carouselStrip) carouselStrip.style.display = 'none';
    }

    // Render Bounding Boxes with relevance-based visual hierarchy
    if (currentSource.components && currentSource.components.length > 0) {
        // Filter by threshold
        const visibleComponents = currentSource.components.filter(c =>
            (c.relevance_score || 0) >= relevanceThreshold
        );

        visibleComponents.forEach((c, i) => {
            if (c.bbox_2d && c.bbox_2d.length === 4) {
                const [x1, y1, x2, y2] = c.bbox_2d;
                const relevanceScore = c.relevance_score || 0;
                const visualTier = getVisualTier(relevanceScore);

                const el = document.createElement('div');
                el.className = 'bbox-overlay';
                el.setAttribute('data-relevance', relevanceScore.toFixed(2));
                el.setAttribute('title', `${c.label || ''} (${Math.round(relevanceScore * 100)}% relevant)`);

                // Convert 0-1000 range to percentage
                el.style.left = (x1 / 10) + '%';
                el.style.top = (y1 / 10) + '%';
                el.style.width = ((x2 - x1) / 10) + '%';
                el.style.height = ((y2 - y1) / 10) + '%';
                el.style.borderColor = visualTier.color;
                el.style.borderWidth = visualTier.border_width + 'px';
                el.style.opacity = '1'; // Always visible in modal
                // Background color handled by CSS hover state

                el.innerHTML = `<span class="bbox-label" style="background:${visualTier.color};color:#1a1a1a;">${c.label || ''}</span>`;
                modalContainer.appendChild(el);
            }
        });
    }

    // Render Legend - show only visible components with toggle functionality
    renderLegend(currentSource);
}

function renderLegend(source) {
    modalLegend.innerHTML = '';
    if (source.components && source.components.length > 0) {
        const visibleComponents = source.components.filter(c =>
            (c.relevance_score || 0) >= relevanceThreshold
        );

        visibleComponents.forEach((c, i) => {
            const relevanceScore = c.relevance_score || 0;
            const visualTier = getVisualTier(relevanceScore);
            const relevancePercent = Math.round(relevanceScore * 100);

            const item = document.createElement('div');
            item.className = 'legend-item';
            item.setAttribute('data-component-index', i);
            item.innerHTML = `
                <div class="legend-color" style="background: ${visualTier.color}"></div>
                <span>${c.label || c.type} <span style="color: var(--text-muted); font-size: 11px;">(${relevancePercent}%)</span></span>
            `;

            // Add click to toggle visibility
            item.onclick = () => toggleComponentVisibility(i);

            modalLegend.appendChild(item);
        });
    }
}

function toggleComponentVisibility(componentIndex) {
    const bbox = modalContainer.querySelectorAll('.bbox-overlay')[componentIndex];
    const legendItem = modalLegend.querySelectorAll('.legend-item')[componentIndex];

    if (bbox && legendItem) {
        bbox.classList.toggle('hidden');
        legendItem.classList.toggle('hidden');
    }
}

function updateCarouselPosition() {
    // Update counter with flash animation
    const counter = modalContainer.querySelector('.carousel-counter');
    if (counter) {
        counter.classList.add('updated');
        counter.textContent = `${carouselState.currentIndex + 1} / ${carouselState.totalImages}`;
        setTimeout(() => counter.classList.remove('updated'), 300);
    }
}

function navigateCarousel(direction) {
    const newIndex = carouselState.currentIndex + direction;

    // Wrap around
    if (newIndex < 0) {
        carouselState.currentIndex = carouselState.totalImages - 1;
    } else if (newIndex >= carouselState.totalImages) {
        carouselState.currentIndex = 0;
    } else {
        carouselState.currentIndex = newIndex;
    }

    // Add slide animation class to image
    modalImage.classList.add(direction > 0 ? 'slide-left' : 'slide-right');

    setTimeout(() => {
        renderCarouselUI();
        updateCarouselPosition();
        modalImage.classList.remove('slide-left', 'slide-right');
    }, 300);
}

function jumpToImage(index) {
    if (index >= 0 && index < carouselState.totalImages) {
        carouselState.currentIndex = index;
        renderCarouselUI();
        updateCarouselPosition();
    }
}

function renderThumbnailStrip() {
    if (!carouselStrip) return;

    carouselStrip.innerHTML = '';
    carouselStrip.style.display = 'flex';

    carouselState.images.forEach((img, index) => {
        const thumb = document.createElement('div');
        thumb.className = 'strip-thumb' + (index === carouselState.currentIndex ? ' active' : '');
        thumb.setAttribute('data-index', index);
        thumb.setAttribute('title', `${index + 1}. ${img.title || 'Image'} (Press ${index + 1})`);

        // Create thumbnail image
        const thumbImg = document.createElement('img');
        thumbImg.src = img.url;
        thumbImg.alt = img.title || `Image ${index + 1}`;
        thumb.appendChild(thumbImg);

        // Add index number overlay
        const indexBadge = document.createElement('span');
        indexBadge.className = 'thumb-index';
        indexBadge.textContent = index + 1;
        thumb.appendChild(indexBadge);

        // Click to jump to image
        thumb.onclick = (e) => {
            e.stopPropagation();
            jumpToImage(index);
        };

        carouselStrip.appendChild(thumb);
    });

    // Scroll active thumbnail into view
    const activeThumb = carouselStrip.querySelector('.strip-thumb.active');
    if (activeThumb) {
        activeThumb.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
}

function closeImageModal() {
    imageModal.style.display = 'none';
    imageModal.classList.remove('active');
}

// Enhanced keyboard navigation for modal and carousel
document.addEventListener('keydown', (e) => {
    // Only handle if modal is active
    if (!imageModal.classList.contains('active')) return;

    switch(e.key) {
        case 'Escape':
            closeImageModal();
            break;
        case 'ArrowLeft':
            if (carouselState.totalImages > 1) {
                e.preventDefault();
                navigateCarousel(-1);
            }
            break;
        case 'ArrowRight':
            if (carouselState.totalImages > 1) {
                e.preventDefault();
                navigateCarousel(1);
            }
            break;
        case '1': case '2': case '3': case '4': case '5':
        case '6': case '7': case '8': case '9':
            // Quick jump to thumbnail (1-indexed)
            const imageIndex = parseInt(e.key) - 1;
            if (imageIndex < carouselState.totalImages) {
                e.preventDefault();
                jumpToImage(imageIndex);
            }
            break;
    }
});

// Close on click outside
imageModal.addEventListener('click', (e) => {
    if (e.target === imageModal) closeImageModal();
});

// Simple markdown parser for basic formatting
function parseMarkdown(text) {
    if (!text) return '';

    // Escape HTML first to prevent XSS
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Parse markdown elements
    // Bold: **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

    // Italic: *text* or _text_
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/_([^_]+)_/g, '<em>$1</em>');

    // Links: [text](url)
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Inline code: `code`
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Convert numbered lists (1. item) - handle multi-line lists properly
    const lines = html.split('\n');
    let inOrderedList = false;
    let inUnorderedList = false;
    const processedLines = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const isNumberedItem = /^\d+\.\s+(.+)$/.test(line);
        const isBulletItem = /^[-*]\s+(.+)$/.test(line);

        if (isNumberedItem) {
            if (!inOrderedList) {
                processedLines.push('<ol>');
                inOrderedList = true;
            }
            if (inUnorderedList) {
                processedLines.push('</ul>');
                inUnorderedList = false;
            }
            processedLines.push(line.replace(/^\d+\.\s+(.+)$/, '<li>$1</li>'));
        } else if (isBulletItem) {
            if (!inUnorderedList) {
                processedLines.push('<ul>');
                inUnorderedList = true;
            }
            if (inOrderedList) {
                processedLines.push('</ol>');
                inOrderedList = false;
            }
            processedLines.push(line.replace(/^[-*]\s+(.+)$/, '<li>$1</li>'));
        } else {
            if (inOrderedList) {
                processedLines.push('</ol>');
                inOrderedList = false;
            }
            if (inUnorderedList) {
                processedLines.push('</ul>');
                inUnorderedList = false;
            }
            processedLines.push(line);
        }
    }

    // Close any open lists
    if (inOrderedList) processedLines.push('</ol>');
    if (inUnorderedList) processedLines.push('</ul>');

    html = processedLines.join('\n');

    // Paragraphs: double newlines
    html = html.replace(/\n\n+/g, '</p><p>');

    // Single newlines within paragraphs (but not in lists)
    html = html.replace(/(?<!<\/li>)\n(?!<[ou]l>)(?!<li>)(?!<\/[ou]l>)/g, '<br>');

    // Wrap in paragraph tags
    html = '<p>' + html + '</p>';

    // Clean up paragraphs around lists
    html = html.replace(/<p>(<[ou]l>)/g, '$1');
    html = html.replace(/(<\/[ou]l>)<\/p>/g, '$1');
    html = html.replace(/<p>\s*<\/p>/g, '');
    html = html.replace(/<p><br>/g, '<p>');
    html = html.replace(/<br><\/p>/g, '</p>');

    return html;
}

// Add message to UI
function addMessage(role, content, imageUrl = null) {
    emptyState.style.display = 'none';
    
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    let imageHtml = '';
    if (imageUrl) {
        imageHtml = `<img src="${imageUrl}" alt="Uploaded image" class="message-image">`;
    }
    
    msg.innerHTML = `
        <div class="message-header">
            <span class="message-role">${role === 'user' ? 'You' : 'Aeris'}</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">
            ${parseMarkdown(content)}
            ${imageHtml}
        </div>
    `;
    
    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return msg;
}

// Add assistant message with metadata
function addAssistantMessage(data) {
    emptyState.style.display = 'none';
    
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Confidence level
    let confClass = 'high';
    let confLabel = 'High confidence';
    if (data.confidence < 0.5) {
        confClass = 'low';
        confLabel = 'Low confidence';
    } else if (data.confidence < 0.7) {
        confClass = 'medium';
        confLabel = 'Medium confidence';
    }
    
    let extraContent = '';
    if (data.escalated) {
        extraContent = `
            <div class="escalated">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z"/>
                </svg>
                Connecting to human agent...
            </div>
        `;
    }
    
    // Check for escalation hint in message and separate it
    let mainMessage = data.message;
    let escalationHint = '';
    if (mainMessage.includes("If this doesn't fully answer your question")) {
        const parts = mainMessage.split(/\n\n_If this doesn't/);
        mainMessage = parts[0];
        if (parts[1]) {
            escalationHint = `<div class="escalation-hint">💬 If this doesn't ${parts[1].replace(/_$/, '')}</div>`;
        }
    }
    
    msg.innerHTML = `
        <div class="message-wrapper">
            <div class="message-header">
                <span class="message-role">Aeris</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-content">
                ${parseMarkdown(mainMessage)}
                ${escalationHint}
                ${extraContent}
                <div class="confidence" title="${confLabel}">
                    <div class="confidence-bar">
                        <div class="confidence-fill ${confClass}" style="width: ${data.confidence * 100}%"></div>
                    </div>
                    <span>${Math.round(data.confidence * 100)}%</span>
                </div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Update sources with the new source_images format
    updateSources(data.sources || [], data.source_images || []);
}

// Update turn counter with animation
function updateTurnCounter(turn) {
    const counter = document.getElementById('turnCounter');
    const numberEl = counter.querySelector('.turn-number');

    if (turn > 0) {
        // Update number with scale animation
        if (numberEl) {
            numberEl.style.transform = 'scale(1.3)';
            setTimeout(() => {
                numberEl.textContent = turn;
                numberEl.style.transform = 'scale(1)';
            }, 150);
        }

        counter.style.display = 'inline-flex';

        // Celebrate milestones (every 5 turns)
        if (turn % 5 === 0) {
            counter.classList.add('milestone');
            setTimeout(() => counter.classList.remove('milestone'), 600);
        }
    }
}

// Show context warning with action button
function showContextWarning(message) {
    const warning = document.getElementById('contextWarning');
    const textEl = warning.querySelector('.warning-text');

    if (textEl) {
        textEl.textContent = message;
    }

    warning.style.display = 'flex';
    // No auto-hide - user must explicitly dismiss or take action
}

// Dismiss context warning
function dismissContextWarning() {
    const warning = document.getElementById('contextWarning');
    warning.style.animation = 'warningSlideOut 0.3s ease-out forwards';

    setTimeout(() => {
        warning.style.display = 'none';
        warning.style.animation = '';
    }, 300);
}

// Start new chat
function startNewChat() {
    if (!confirm('Start a new conversation? Your current context will be cleared.')) {
        return;
    }

    // Reset state
    conversationId = null;
    currentTurn = 0;
    currentImage = null;

    // Clear UI with new empty state
    messagesContainer.innerHTML = `
        <div class="empty-state" id="emptyState">
            <div class="aeris-avatar">
                <div class="avatar-icon"></div>
            </div>
            <h2 class="empty-title">Hi! I'm Aeris, your knowledge assistant</h2>
            <p class="empty-subtitle">
                I can help you find information from your uploaded manuals and documents.
                Ask questions in plain language or upload images to search visually.
            </p>
            <div class="capability-pills" id="capabilityPills"></div>
            <div class="suggestions-section">
                <div class="suggestions-label">Try asking about:</div>
                <div class="suggestions"></div>
            </div>
        </div>
    `;

    // Populate suggestions and capabilities
    populateEmptyState();

    const turnCounter = document.getElementById('turnCounter');
    if (turnCounter) turnCounter.style.display = 'none';

    // Dismiss warning with animation
    dismissContextWarning();
    sourcesList.innerHTML = `
        <div class="sources-empty">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <path d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
            </svg>
            <p>Sources will appear here when you ask a question</p>
        </div>
    `;

    clearImagePreview();
    messageInput.value = '';
    messageInput.focus();

    attachSuggestionListeners();
}

// Update sources panel with visual grounding support
function updateSources(sources, sourceImages) {
    // Store current source images for threshold updates
    currentSourceImages = sourceImages.length > 0 ? sourceImages : sources.map(s => ({
        url: s.url,
        title: s.title,
        caption: '',
        components: [],
        match_type: 'semantic',
        score: s.relevance_score,
        type: s.type
    }));

    if (!sources.length && !sourceImages.length) {
        sourcesList.innerHTML = `
            <div class="sources-empty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"/>
                </svg>
                <p>Sources will appear here when you ask a question</p>
            </div>
        `;
        return;
    }

    sourcesList.innerHTML = '';

    const displaySources = currentSourceImages;

    displaySources.forEach((source, index) => {
        const card = document.createElement('div');
        card.className = 'source-card';
        card.style.animationDelay = `${index * 0.1}s`;
        
        // Build image HTML with visual grounding overlay and bounding boxes
        let imageHtml = '';
        if (source.url) {
            // Build grounding tags and bounding box overlays from components
            let groundingHtml = '';
            let bboxOverlays = '';
            if (source.components && source.components.length > 0) {
                // Filter components by threshold
                const visibleComponents = source.components.filter(c =>
                    (c.relevance_score || 0) >= relevanceThreshold
                );

                if (visibleComponents.length > 0) {
                    const tags = visibleComponents.slice(0, 4).map((c, i) => {
                        const relevanceScore = c.relevance_score || 0;
                        const relevancePercent = Math.round(relevanceScore * 100);
                        return `<span class="grounding-tag" data-bbox-index="${i}" title="${relevancePercent}% relevant">${c.label || c.type || 'Component'}</span>`;
                    }).join('');
                    groundingHtml = `<div class="source-grounding">${tags}</div>`;

                    // Create bounding box overlays with relevance-based styling
                    bboxOverlays = visibleComponents.map((c, i) => {
                        if (c.bbox_2d && c.bbox_2d.length === 4) {
                            const [x1, y1, x2, y2] = c.bbox_2d;
                            const relevanceScore = c.relevance_score || 0;
                            const visualTier = getVisualTier(relevanceScore);

                            // Convert 0-1000 to percentage for CSS positioning
                            const left = (x1 / 10);
                            const top = (y1 / 10);
                            const width = ((x2 - x1) / 10);
                            const height = ((y2 - y1) / 10);

                            return `<div class="bbox-overlay" data-bbox-index="${i}" data-relevance="${relevanceScore.toFixed(2)}"
                                style="left:${left}%;top:${top}%;width:${width}%;height:${height}%;
                                       border-color:${visualTier.color};border-width:${visualTier.border_width}px;
                                       opacity:${visualTier.opacity};background-color:${visualTier.color}20;">
                                <span class="bbox-label"
                                      style="background:${visualTier.color};color:#1a1a1a;
                                             font-size:${visualTier.font_size};font-weight:${visualTier.font_weight};">
                                    ${c.label || ''}
                                </span>
                            </div>`;
                        }
                        return '';
                    }).join('');
                }
            }
            
            imageHtml = `
                <div class="source-image-container" data-source-index="${index}">
                    <img src="${source.url}" alt="${source.title}" class="source-image">
                    ${bboxOverlays}
                    ${groundingHtml}
                </div>
            `;
        }
        
        // Type icon
        const typeIcon = source.type === 'image' || source.url
            ? '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>'
            : '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>';
        
        // Match type badge
        const matchType = source.match_type || 'semantic';
        const matchLabel = {
            'exact': 'Exact',
            'visual': 'Visual',
            'textual': 'Text',
            'semantic': 'Semantic'
        }[matchType] || 'Match';
        
        // Caption (truncated)
        const captionHtml = source.caption 
            ? `<div class="source-caption">${source.caption}</div>`
            : '';
        
        // Score as percentage with visual bar
        const scorePercent = Math.round((source.score || 0) * 100);
        
        card.innerHTML = `
            ${imageHtml}
            <div class="source-content">
                <div class="source-type">
                    ${typeIcon}
                    <span>IMAGE</span>
                    <span class="source-match-type">${matchLabel}</span>
                </div>
                <div class="source-title">${source.title}</div>
                ${captionHtml}
                <div class="source-score">
                    <div class="source-score-bar">
                        <div class="source-score-fill" style="width: ${scorePercent}%"></div>
                    </div>
                    <span>${scorePercent}%</span>
                </div>
            </div>
        `;
        
        if (source.url) {
            card.style.cursor = 'pointer';
            // Pass ALL source images for carousel navigation, with current index
            // This enables prev/next navigation in the modal
            card.addEventListener('click', () => {
                // Filter to only images with URLs for carousel
                const allImages = displaySources.filter(s => s.url);
                openImageModal(allImages, index);
            });
        }

        sourcesList.appendChild(card);
    });
}

// Create streaming assistant message container
function createStreamingAssistantMessage() {
    emptyState.style.display = 'none';

    const msg = document.createElement('div');
    msg.className = 'message assistant';

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    msg.innerHTML = `
        <div class="message-header">
            <span class="message-role">Aeris</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">
            <p class="streaming-content"></p>
            <span class="streaming-cursor">▋</span>
        </div>
    `;

    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    return msg;
}

// Update streaming message content
function updateStreamingMessageContent(msgElement, text) {
    const contentDiv = msgElement.querySelector('.streaming-content');
    if (contentDiv) {
        contentDiv.innerHTML = parseMarkdown(text);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

// Update streaming message with final metadata
function updateStreamingMessage(msgElement, text, metadata) {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Confidence level
    let confClass = 'high';
    let confLabel = 'High confidence';
    if (metadata.confidence < 0.5) {
        confClass = 'low';
        confLabel = 'Low confidence';
    } else if (metadata.confidence < 0.7) {
        confClass = 'medium';
        confLabel = 'Medium confidence';
    }

    // Check for escalation hint in message
    let mainMessage = text;
    let escalationHint = '';
    if (mainMessage.includes("If this doesn't fully answer your question")) {
        const parts = mainMessage.split(/\n\n_If this doesn't/);
        mainMessage = parts[0];
        if (parts[1]) {
            escalationHint = `<div class="escalation-hint">💬 If this doesn't ${parts[1].replace(/_$/, '')}</div>`;
        }
    }

    msgElement.innerHTML = `
        <div class="message-header">
            <span class="message-role">Aeris</span>
            <span class="message-time">${time}</span>
        </div>
        <div class="message-content">
            ${parseMarkdown(mainMessage)}
            ${escalationHint}
            <div class="confidence" title="${confLabel}">
                <div class="confidence-bar">
                    <div class="confidence-fill ${confClass}" style="width: ${metadata.confidence * 100}%"></div>
                </div>
                <span>${Math.round(metadata.confidence * 100)}%</span>
            </div>
        </div>
    `;

    // Update sources
    updateSources(metadata.sources || [], metadata.source_images || []);
}

// Send message with streaming
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message && !currentImage) return;
    if (isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;

    // OPTIMISTIC UI: Show user message immediately (< 50ms)
    let imageDataUrl = null;
    if (currentImage) {
        imageDataUrl = previewImage.src;
    }
    addMessage('user', message, imageDataUrl);

    // Clear input immediately
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show typing indicator while processing
    const typingMsg = document.createElement('div');
    typingMsg.className = 'typing';
    typingMsg.id = 'typingIndicator';
    typingMsg.innerHTML = `
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
        <span>Searching knowledge base...</span>
    `;
    messagesContainer.appendChild(typingMsg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        // Prepare form data for streaming endpoint
        const formData = new FormData();
        formData.append('message', message);
        if (currentImage) {
            formData.append('image', currentImage);
        }
        if (conversationId) {
            formData.append('conversation_id', conversationId);
        }

        // Use fetch with streaming
        const response = await fetch(`${API_BASE}/chat/stream`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        // Remove typing indicator
        const typingEl = document.getElementById('typingIndicator');
        if (typingEl) typingEl.remove();

        // Create assistant message container for streaming
        const assistantMsg = createStreamingAssistantMessage();

        // Read the stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullText = '';
        let metadata = null;

        while (true) {
            const { done, value } = await reader.read();

            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete lines
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (!line.trim() || !line.startsWith('data: ')) continue;

                const data = line.substring(6); // Remove 'data: ' prefix

                if (data === '[DONE]') {
                    // Stream complete
                    if (metadata) {
                        updateStreamingMessage(assistantMsg, fullText, metadata);
                    }
                    continue;
                }

                try {
                    const parsed = JSON.parse(data);

                    // EARLY METADATA: Received immediately before streaming
                    if (parsed.type === 'metadata') {
                        metadata = parsed;
                        conversationId = parsed.conversation_id;
                        currentTurn = parsed.turn || 0;
                        updateTurnCounter(currentTurn);
                    }

                    if (parsed.type === 'token') {
                        fullText += parsed.content;
                        updateStreamingMessageContent(assistantMsg, fullText);
                    } else if (parsed.type === 'error') {
                        fullText += '\n\n' + parsed.content;
                        updateStreamingMessageContent(assistantMsg, fullText);
                    }
                } catch (e) {
                    console.error('Error parsing stream data:', e, data);
                }
            }
        }

        // Final update with metadata
        if (metadata) {
            updateStreamingMessage(assistantMsg, fullText, metadata);

            // Show context warning if present
            if (metadata.context_warning) {
                showContextWarning(metadata.context_warning);
            }
        } else {
            // Fallback if metadata never arrived
            console.warn('No metadata received, using defaults');
            const fallbackMetadata = {
                sources: [],
                source_images: [],
                confidence: 0.0,
                latency_ms: 0
            };
            updateStreamingMessage(assistantMsg, fullText || 'No response received.', fallbackMetadata);
        }

    } catch (error) {
        console.error('Error:', error);
        const typingEl = document.getElementById('typingIndicator');
        if (typingEl) typingEl.remove();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}. Please try again.`);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        clearImagePreview();
    }
}

// General Keyboard Shortcuts (non-modal)
document.addEventListener('keydown', (e) => {
    // Skip if modal is active (handled by modal-specific handler)
    if (imageModal && imageModal.classList.contains('active')) return;

    // Cmd/Ctrl + Enter = Send message
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        sendMessage();
    }

    // Cmd/Ctrl + Shift + N = New chat
    if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'N') {
        e.preventDefault();
        startNewChat();
    }

    // Cmd/Ctrl + U = Upload image
    if ((e.metaKey || e.ctrlKey) && e.key === 'u') {
        e.preventDefault();
        document.getElementById('fileInput').click();
    }
});

// Update relevance threshold and re-render sources
function updateRelevanceThreshold(newThreshold) {
    relevanceThreshold = newThreshold;

    // Update slider display
    const thresholdSlider = document.getElementById('relevanceThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    if (thresholdSlider) thresholdSlider.value = Math.round(newThreshold * 100);
    if (thresholdValue) thresholdValue.textContent = Math.round(newThreshold * 100);

    // Re-render sources if they exist
    if (currentSourceImages && currentSourceImages.length > 0) {
        updateSources([], currentSourceImages);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Load UX configuration
    loadUXConfig();

    messageInput.focus();

    // Threshold slider control (if exists)
    const thresholdSlider = document.getElementById('relevanceThreshold');
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', (e) => {
            updateRelevanceThreshold(parseInt(e.target.value) / 100);
        });
    }

    // Preset buttons (if exist)
    const showAllBtn = document.getElementById('showAllBtn');
    const showRelevantBtn = document.getElementById('showRelevantBtn');
    const showHighlyRelevantBtn = document.getElementById('showHighlyRelevantBtn');

    if (showAllBtn) {
        showAllBtn.addEventListener('click', () => updateRelevanceThreshold(0));
    }
    if (showRelevantBtn) {
        showRelevantBtn.addEventListener('click', () => updateRelevanceThreshold(0.25));
    }
    if (showHighlyRelevantBtn) {
        showHighlyRelevantBtn.addEventListener('click', () => updateRelevanceThreshold(0.5));
    }

    // Add event delegation for grounding tag hover interactions
    // When hovering a grounding tag, highlight the corresponding bounding box
    document.getElementById('sourcesList').addEventListener('mouseover', (e) => {
        const tag = e.target.closest('.grounding-tag');
        if (tag) {
            const bboxIndex = tag.dataset.bboxIndex;
            const container = tag.closest('.source-image-container');
            if (container && bboxIndex !== undefined) {
                const bbox = container.querySelector(`.bbox-overlay[data-bbox-index="${bboxIndex}"]`);
                if (bbox) {
                    bbox.style.opacity = '1';
                    bbox.style.borderWidth = '3px';
                    bbox.style.boxShadow = '0 0 10px rgba(212, 165, 116, 0.5)';
                }
            }
        }
    });
    
    document.getElementById('sourcesList').addEventListener('mouseout', (e) => {
        const tag = e.target.closest('.grounding-tag');
        if (tag) {
            const bboxIndex = tag.dataset.bboxIndex;
            const container = tag.closest('.source-image-container');
            if (container && bboxIndex !== undefined) {
                const bbox = container.querySelector(`.bbox-overlay[data-bbox-index="${bboxIndex}"]`);
                if (bbox) {
                    bbox.style.opacity = '';
                    bbox.style.borderWidth = '';
                    bbox.style.boxShadow = '';
                }
            }
        }
    });
});