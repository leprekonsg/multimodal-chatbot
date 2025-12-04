const API_BASE = window.location.origin;

// State
let uploadQueue = [];
let recentDocs = [];

// DOM
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const queueSection = document.getElementById('queue');
const queueItems = document.getElementById('queueItems');
const textTitle = document.getElementById('textTitle');
const textContent = document.getElementById('textContent');
const submitText = document.getElementById('submitText');
const recentSection = document.getElementById('recentSection');
const recentGrid = document.getElementById('recentGrid');

// File type icons
const icons = {
    pdf: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>`,
    image: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>`,
    text: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8"/></svg>`
};

// Get file type
function getFileType(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    if (ext === 'pdf') return 'pdf';
    if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) return 'image';
    return 'text';
}

// Format file size
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Toast notification
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            ${type === 'success' 
                ? '<path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/>'
                : '<circle cx="12" cy="12" r="10"/><path d="M15 9l-6 6M9 9l6 6"/>'}
        </svg>
        ${message}
    `;
    document.getElementById('toastContainer').appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// Upload zone events
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    fileInput.value = '';
});

// Handle file selection
function handleFiles(files) {
    for (const file of files) {
        addToQueue(file);
    }
    processQueue();
}

// Add file to queue
function addToQueue(file) {
    const id = Date.now() + Math.random();
    const item = {
        id,
        file,
        status: 'pending',
        progress: 0,
        message: 'Waiting...'
    };
    uploadQueue.push(item);
    renderQueue();
}

// Render queue
function renderQueue() {
    if (uploadQueue.length === 0) {
        queueSection.style.display = 'none';
        return;
    }

    queueSection.style.display = 'block';
    queueItems.innerHTML = uploadQueue.map(item => {
        const type = getFileType(item.file.name);
        const statusClass = `status-${item.status}`;

        return `
            <div class="queue-item" data-id="${item.id}">
                <div class="queue-item-icon ${type}">${icons[type]}</div>
                <div class="queue-item-info">
                    <div class="queue-item-name">${item.file.name}</div>
                    <div class="queue-item-meta">${formatSize(item.file.size)}</div>
                    ${item.status === 'processing' ? `
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${item.progress}%"></div>
                        </div>
                    ` : ''}
                </div>
                <div class="queue-item-status ${statusClass}">
                    ${item.status === 'processing' ? '<div class="spinner"></div>' : ''}
                    <span class="status-text">${item.message || item.status}</span>
                </div>
                ${item.status === 'processing' ? `
                    <button class="queue-item-cancel" onclick="cancelIngestion(${item.id})" title="Stop after current page">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="6" y="6" width="12" height="12" rx="1"/>
                        </svg>
                    </button>
                ` : ''}
                ${item.status === 'done' || item.status === 'error' || item.status === 'cancelled' ? `
                    <button class="queue-item-remove" onclick="removeFromQueue(${item.id})">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M6 18L18 6M6 6l12 12"/>
                        </svg>
                    </button>
                ` : ''}
            </div>
        `;
    }).join('');
}

// Remove from queue
function removeFromQueue(id) {
    uploadQueue = uploadQueue.filter(item => item.id !== id);
    renderQueue();
}

// Cancel ongoing ingestion
function cancelIngestion(id) {
    const item = uploadQueue.find(i => i.id === id);
    if (item && item.abortController) {
        item.abortController.abort();
        item.message = 'Stopping after current page...';
        renderQueue();
    }
}

// Update specific item in queue (avoids full re-render)
function updateItemProgress(id, progress, message) {
    const item = uploadQueue.find(i => i.id === id);
    if (item) {
        item.progress = progress;
        item.message = message;
        
        const el = document.querySelector(`.queue-item[data-id="${id}"]`);
        if (el) {
            const fill = el.querySelector('.progress-fill');
            const text = el.querySelector('.status-text');
            if (fill) fill.style.width = `${progress}%`;
            if (text) text.textContent = message;
        }
    }
}

// Process queue with Streaming Response support
async function processQueue() {
    const pending = uploadQueue.find(item => item.status === 'pending');
    if (!pending) return;

    pending.status = 'processing';
    pending.message = 'Uploading...';
    pending.abortController = new AbortController(); // Add abort controller
    renderQueue();

    try {
        const formData = new FormData();
        formData.append('file', pending.file);

        // Use fetch with stream reader and abort signal
        const response = await fetch(`${API_BASE}/ingest/file`, {
            method: 'POST',
            body: formData,
            signal: pending.abortController.signal // Add abort signal
        });

        if (!response.ok) throw new Error(`Server Error: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');

            // Process all complete lines
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const data = JSON.parse(line);

                    if (data.type === 'progress') {
                        // Update progress bar
                        // Map 0-100% processing to 20-100% total (first 20% is upload)
                        const totalProgress = 20 + (data.value * 0.8);
                        updateItemProgress(pending.id, totalProgress, data.message);
                    } else if (data.type === 'complete') {
                        // Final result
                        pending.status = 'done';
                        pending.message = 'Complete';
                        pending.progress = 100;

                        // Add to recent
                        if (data.docs) {
                            data.docs.forEach(doc => {
                                recentDocs.unshift({
                                    ...doc,
                                    filename: pending.file.name
                                });
                            });
                            renderRecent();
                        }
                    } else if (data.type === 'cancelled') {
                        // User cancelled ingestion
                        pending.status = 'cancelled';
                        pending.message = `Stopped (${data.pages_completed || 0} pages saved)`;
                        pending.progress = 100;
                        showToast(`${pending.file.name}: ${data.message}`, 'warning');
                    } else if (data.type === 'error') {
                        throw new Error(data.message);
                    }
                } catch (e) {
                    console.log('Non-JSON chunk:', line);
                }
            }
        }

        // If we finished stream without explicit complete message (legacy backend support)
        if (pending.status === 'processing') {
            pending.status = 'done';
            pending.message = 'Complete';
            pending.progress = 100;
        }

        renderQueue();
        if (pending.status === 'done') {
            showToast(`${pending.file.name} ingested successfully`);
        }

    } catch (error) {
        // Check if error was due to abort
        if (error.name === 'AbortError') {
            pending.status = 'cancelled';
            pending.message = 'Cancelled by user';
            renderQueue();
            showToast(`${pending.file.name} ingestion cancelled`, 'warning');
        } else {
            console.error(error);
            pending.status = 'error';
            pending.message = error.message || 'Failed';
            renderQueue();
            showToast(`Failed to ingest ${pending.file.name}`, 'error');
        }
    }

    processQueue(); // Process next
}

// Text input
textContent.addEventListener('input', () => {
    submitText.disabled = !textContent.value.trim();
});

submitText.addEventListener('click', async () => {
    const content = textContent.value.trim();
    if (!content) return;
    
    submitText.disabled = true;
    submitText.textContent = 'Adding...';
    
    try {
        const response = await fetch(`${API_BASE}/ingest/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: content,
                title: textTitle.value.trim() || null
            })
        });
        
        if (!response.ok) throw new Error('Failed to add text');
        
        const doc = await response.json();
        recentDocs.unshift({
            ...doc,
            filename: textTitle.value.trim() || 'Text Document'
        });
        renderRecent();
        
        textContent.value = '';
        textTitle.value = '';
        showToast('Text added to knowledge base');
        
    } catch (error) {
        showToast('Failed to add text', 'error');
    }
    
    submitText.disabled = true;
    submitText.textContent = 'Add Text';
});

// Render recent documents
function renderRecent() {
    if (recentDocs.length === 0) {
        recentSection.style.display = 'none';
        return;
    }
    
    recentSection.style.display = 'block';
    recentGrid.innerHTML = recentDocs.slice(0, 6).map(doc => `
        <div class="recent-card">
            ${doc.url ? `<img src="${doc.url}" class="recent-card-image" alt="">` : ''}
            <div class="recent-card-content">
                <div class="recent-card-type">${doc.type}</div>
                <div class="recent-card-title">${doc.filename || doc.id.slice(0, 8)}</div>
                ${doc.caption ? `<div class="recent-card-caption">${doc.caption}</div>` : ''}
            </div>
        </div>
    `).join('');
}