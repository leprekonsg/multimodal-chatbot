// State
let conversationId = null;
let currentImage = null;
let isLoading = false;

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

// Suggestion buttons
document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', function() {
        messageInput.value = this.dataset.query;
        sendMessage();
    });
});

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

// Modal Functions
function openImageModal(source) {
    modalTitle.textContent = source.title || 'Image View';
    modalImage.src = source.url;
    
    // Clear previous overlays (keep the image)
    const existingOverlays = modalContainer.querySelectorAll('.bbox-overlay');
    existingOverlays.forEach(el => el.remove());

    // Render Bounding Boxes
    if (source.components && source.components.length > 0) {
        source.components.forEach((c, i) => {
            if (c.bbox_2d && c.bbox_2d.length === 4) {
                const [x1, y1, x2, y2] = c.bbox_2d;
                const colors = ['#D4A574', '#6B8B6B', '#C17B7B', '#7B9BC1', '#B87BC1'];
                const color = colors[i % colors.length];
                
                const el = document.createElement('div');
                el.className = 'bbox-overlay';
                // Convert 0-1000 range to percentage
                el.style.left = (x1 / 10) + '%';
                el.style.top = (y1 / 10) + '%';
                el.style.width = ((x2 - x1) / 10) + '%';
                el.style.height = ((y2 - y1) / 10) + '%';
                el.style.borderColor = color;
                el.style.opacity = '1'; // Always visible in modal
                
                el.innerHTML = `<span class="bbox-label" style="background:${color}">${c.label || ''}</span>`;
                modalContainer.appendChild(el);
            }
        });
    }
    
    // Render Legend
    modalLegend.innerHTML = '';
    if (source.components && source.components.length > 0) {
        source.components.forEach((c, i) => {
            const colors = ['#D4A574', '#6B8B6B', '#C17B7B', '#7B9BC1', '#B87BC1'];
            const color = colors[i % colors.length];
            
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="legend-color" style="background: ${color}"></div>
                <span>${c.label || c.type}</span>
            `;
            modalLegend.appendChild(item);
        });
    }

    imageModal.classList.add('active');
    imageModal.style.display = 'flex';
}

function closeImageModal() {
    imageModal.style.display = 'none';
    imageModal.classList.remove('active');
}

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeImageModal();
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
    
    // Convert numbered lists (1. item)
    html = html.replace(/^(\d+)\.\s+(.+)$/gm, '<li>$2</li>');
    // Wrap consecutive li elements in ol
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ol>$&</ol>');
    
    // Convert bullet lists (- item or * item)
    html = html.replace(/^[-*]\s+(.+)$/gm, '<li>$1</li>');
    
    // Paragraphs: double newlines
    html = html.replace(/\n\n+/g, '</p><p>');
    
    // Single newlines within paragraphs
    html = html.replace(/\n/g, '<br>');
    
    // Wrap in paragraph tags
    html = '<p>' + html + '</p>';
    
    // Clean up empty paragraphs
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
            <span class="message-role">${role === 'user' ? 'You' : 'Console'}</span>
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
        <div class="message-header">
            <span class="message-role">Console</span>
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
    `;
    
    messagesContainer.appendChild(msg);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Update sources with the new source_images format
    updateSources(data.sources || [], data.source_images || []);
}

// Show typing indicator
function showTyping() {
    const typing = document.createElement('div');
    typing.className = 'typing';
    typing.id = 'typingIndicator';
    typing.innerHTML = `
        <div class="typing-dots">
            <span></span><span></span><span></span>
        </div>
        <span>Searching knowledge base...</span>
    `;
    messagesContainer.appendChild(typing);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

// Update sources panel with visual grounding support
function updateSources(sources, sourceImages) {
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
    
    // Use sourceImages directly as they now contain all the data we need
    const displaySources = sourceImages.length > 0 ? sourceImages : sources.map(s => ({
        url: s.url,
        title: s.title,
        caption: '',
        components: [],
        match_type: 'semantic',
        score: s.relevance_score,
        type: s.type
    }));
    
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
                const tags = source.components.slice(0, 4).map((c, i) => 
                    `<span class="grounding-tag" data-bbox-index="${i}">${c.label || c.type || 'Component'}</span>`
                ).join('');
                groundingHtml = `<div class="source-grounding">${tags}</div>`;
                
                // Create bounding box overlays (using 0-1000 normalized coords)
                // These will be drawn via CSS positioning
                bboxOverlays = source.components.map((c, i) => {
                    if (c.bbox_2d && c.bbox_2d.length === 4) {
                        const [x1, y1, x2, y2] = c.bbox_2d;
                        // Convert 0-1000 to percentage for CSS positioning
                        const left = (x1 / 10);
                        const top = (y1 / 10);
                        const width = ((x2 - x1) / 10);
                        const height = ((y2 - y1) / 10);
                        const colors = ['#D4A574', '#6B8B6B', '#C17B7B', '#7B9BC1', '#B87BC1'];
                        const color = colors[i % colors.length];
                        return `<div class="bbox-overlay" data-bbox-index="${i}" 
                            style="left:${left}%;top:${top}%;width:${width}%;height:${height}%;border-color:${color};">
                            <span class="bbox-label" style="background:${color}">${c.label || ''}</span>
                        </div>`;
                    }
                    return '';
                }).join('');
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
            // Pass the full source object to the modal function
            // We need to attach the event listener properly since we're creating elements dynamically
            card.addEventListener('click', () => openImageModal(source));
        }
        
        sourcesList.appendChild(card);
    });
}

// Send message
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message && !currentImage) return;
    if (isLoading) return;
    
    isLoading = true;
    sendBtn.disabled = true;
    
    // Show user message
    let imageDataUrl = null;
    if (currentImage) {
        imageDataUrl = previewImage.src;
    }
    addMessage('user', message, imageDataUrl);
    
    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Show typing
    showTyping();
    
    try {
        let response;
        
        if (currentImage) {
            // Multimodal request
            const formData = new FormData();
            formData.append('message', message);
            formData.append('image', currentImage);
            if (conversationId) {
                formData.append('conversation_id', conversationId);
            }
            
            response = await fetch(`${API_BASE}/chat/multimodal`, {
                method: 'POST',
                body: formData
            });
        } else {
            // Text-only request
            response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: conversationId
                })
            });
        }
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        conversationId = data.conversation_id;
        
        hideTyping();
        addAssistantMessage(data);
        
    } catch (error) {
        console.error('Error:', error);
        hideTyping();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}. Please try again.`);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        clearImagePreview();
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    messageInput.focus();
    
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