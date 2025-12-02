# Frontend Component Implementation Guide
## 4-Tier Gradual Visual Hierarchy for Bounding Boxes

This guide shows how to implement the 4-tier gradual visual hierarchy for component bounding boxes using the `relevance_score` field provided by the backend.

---

## Backend API Changes

### New Endpoint: `/config/ux`

**GET** `http://localhost:8000/config/ux`

Returns configuration for component relevance display:

```json
{
  "component_relevance_tiers": {
    "highly_relevant": 0.75,
    "relevant": 0.50,
    "somewhat_relevant": 0.25,
    "low_relevant": 0.0
  },
  "component_visual_tiers": {
    "highly_relevant": {
      "opacity": 1.0,
      "color": "#2563eb",
      "border_width": 3,
      "font_size": "13px",
      "font_weight": "700"
    },
    "relevant": {
      "opacity": 0.9,
      "color": "#3b82f6",
      "border_width": 2,
      "font_size": "12px",
      "font_weight": "600"
    },
    "somewhat_relevant": {
      "opacity": 0.6,
      "color": "#94a3b8",
      "border_width": 1,
      "font_size": "11px",
      "font_weight": "400"
    },
    "low_relevant": {
      "opacity": 0.3,
      "color": "#9ca3af",
      "border_width": 1,
      "font_size": "10px",
      "font_weight": "300"
    }
  }
}
```

### Updated Response: Components Now Include `relevance_score`

Each component in `/chat` response now includes:

```json
{
  "label": "diagram of aft fuselage tiedown ring",
  "bbox_2d": [510, 63, 704, 259],
  "relevance_score": 0.25,
  "type": "diagram"
}
```

`relevance_score` is a float 0.0-1.0 representing fraction of query tokens matched.

---

## Implementation Option 1: Vanilla JavaScript

### HTML Structure
```html
<div class="source-image-container" id="source-images">
  <!-- Images and bounding boxes will be rendered here -->
</div>

<div class="component-controls">
  <label>
    Show components with relevance ≥ <span id="threshold-value">25</span>%
  </label>
  <input
    type="range"
    id="relevance-threshold"
    min="0"
    max="100"
    value="25"
  />
  <button id="show-all-btn">Show All Components</button>
  <button id="show-relevant-btn">Show Only Relevant</button>
</div>
```

### JavaScript Implementation
```javascript
// Fetch UX config on page load
let uxConfig = null;

async function loadUXConfig() {
  const response = await fetch('/config/ux');
  uxConfig = await response.json();
  console.log('UX Config loaded:', uxConfig);
}

// Determine visual tier based on relevance score
function getVisualTier(relevanceScore) {
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

// Render components with gradual visual hierarchy
function renderComponents(sourceImages) {
  const container = document.getElementById('source-images');
  container.innerHTML = '';

  sourceImages.forEach((sourceImage, index) => {
    // Create image container
    const imgContainer = document.createElement('div');
    imgContainer.className = 'image-container';
    imgContainer.style.position = 'relative';

    // Add image
    const img = document.createElement('img');
    img.src = sourceImage.url;
    img.alt = sourceImage.title;
    img.style.width = '100%';
    img.style.height = 'auto';
    imgContainer.appendChild(img);

    // Render components (bounding boxes)
    if (sourceImage.components && sourceImage.components.length > 0) {
      sourceImage.components.forEach((comp, i) => {
        const relevanceScore = comp.relevance_score || 0;
        const visualTier = getVisualTier(relevanceScore);

        // Check threshold
        const threshold = parseInt(document.getElementById('relevance-threshold').value) / 100;
        if (relevanceScore < threshold) {
          return; // Skip this component
        }

        // Create bounding box
        const bbox = document.createElement('div');
        bbox.className = 'bounding-box';
        bbox.setAttribute('data-relevance', relevanceScore.toFixed(2));
        bbox.setAttribute('title', `${comp.label} (${Math.round(relevanceScore * 100)}% relevant)`);

        // Apply visual tier styling
        bbox.style.position = 'absolute';
        bbox.style.left = `${comp.bbox_2d[0] / 10}%`;
        bbox.style.top = `${comp.bbox_2d[1] / 10}%`;
        bbox.style.width = `${(comp.bbox_2d[2] - comp.bbox_2d[0]) / 10}%`;
        bbox.style.height = `${(comp.bbox_2d[3] - comp.bbox_2d[1]) / 10}%`;
        bbox.style.border = `${visualTier.border_width}px solid ${visualTier.color}`;
        bbox.style.opacity = visualTier.opacity;
        bbox.style.backgroundColor = `${visualTier.color}20`; // 20 = hex for ~12% opacity
        bbox.style.boxSizing = 'border-box';
        bbox.style.pointerEvents = 'none';
        bbox.style.transition = 'opacity 0.2s ease, border-color 0.2s ease';

        // Add label
        const label = document.createElement('div');
        label.className = 'component-label';
        label.textContent = comp.label;
        label.style.position = 'absolute';
        label.style.top = '-22px';
        label.style.left = '0';
        label.style.fontSize = visualTier.font_size;
        label.style.fontWeight = visualTier.font_weight;
        label.style.color = visualTier.color;
        label.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
        label.style.padding = '2px 6px';
        label.style.borderRadius = '3px';
        label.style.whiteSpace = 'nowrap';
        label.style.maxWidth = '200px';
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';

        bbox.appendChild(label);
        imgContainer.appendChild(bbox);
      });
    }

    container.appendChild(imgContainer);
  });
}

// Threshold slider control
document.getElementById('relevance-threshold').addEventListener('input', (e) => {
  const value = e.target.value;
  document.getElementById('threshold-value').textContent = value;
  // Re-render with new threshold
  if (window.currentSourceImages) {
    renderComponents(window.currentSourceImages);
  }
});

// Quick preset buttons
document.getElementById('show-all-btn').addEventListener('click', () => {
  document.getElementById('relevance-threshold').value = 0;
  document.getElementById('threshold-value').textContent = '0';
  if (window.currentSourceImages) {
    renderComponents(window.currentSourceImages);
  }
});

document.getElementById('show-relevant-btn').addEventListener('click', () => {
  document.getElementById('relevance-threshold').value = 25;
  document.getElementById('threshold-value').textContent = '25';
  if (window.currentSourceImages) {
    renderComponents(window.currentSourceImages);
  }
});

// Load config on page load
loadUXConfig();

// Store source images globally for threshold updates
window.currentSourceImages = null;

// When receiving chat response
async function handleChatResponse(response) {
  window.currentSourceImages = response.source_images;
  renderComponents(response.source_images);
}
```

---

## Implementation Option 2: React Component

### TypeScript Interfaces
```typescript
interface Component {
  label: string;
  bbox_2d: [number, number, number, number];
  relevance_score?: number;
  type?: string;
}

interface SourceImage {
  url: string;
  title: string;
  caption: string;
  components: Component[];
  match_type: string;
  score: number;
}

interface VisualTier {
  opacity: number;
  color: string;
  border_width: number;
  font_size: string;
  font_weight: string;
}

interface UXConfig {
  component_relevance_tiers: {
    highly_relevant: number;
    relevant: number;
    somewhat_relevant: number;
    low_relevant: number;
  };
  component_visual_tiers: {
    highly_relevant: VisualTier;
    relevant: VisualTier;
    somewhat_relevant: VisualTier;
    low_relevant: VisualTier;
  };
}
```

### React Component
```typescript
import React, { useState, useEffect } from 'react';

const ComponentRenderer: React.FC<{ sourceImages: SourceImage[] }> = ({ sourceImages }) => {
  const [uxConfig, setUxConfig] = useState<UXConfig | null>(null);
  const [threshold, setThreshold] = useState(0.25); // Default 25%

  // Load UX config on mount
  useEffect(() => {
    fetch('/config/ux')
      .then(res => res.json())
      .then(config => setUxConfig(config))
      .catch(err => console.error('Failed to load UX config:', err));
  }, []);

  // Determine visual tier based on relevance score
  const getVisualTier = (relevanceScore: number): VisualTier => {
    if (!uxConfig) return {
      opacity: 0.5,
      color: '#9ca3af',
      border_width: 1,
      font_size: '11px',
      font_weight: '400'
    };

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
  };

  return (
    <div className="component-renderer">
      {/* Threshold Controls */}
      <div className="controls">
        <label>
          Show components with relevance ≥ {Math.round(threshold * 100)}%
        </label>
        <input
          type="range"
          min="0"
          max="100"
          value={threshold * 100}
          onChange={(e) => setThreshold(parseInt(e.target.value) / 100)}
          className="threshold-slider"
        />
        <div className="preset-buttons">
          <button onClick={() => setThreshold(0)}>Show All</button>
          <button onClick={() => setThreshold(0.25)}>Show Relevant</button>
          <button onClick={() => setThreshold(0.5)}>Show Highly Relevant</button>
        </div>
      </div>

      {/* Source Images with Components */}
      <div className="source-images">
        {sourceImages.map((sourceImage, index) => (
          <div key={index} className="image-container" style={{ position: 'relative' }}>
            <img
              src={sourceImage.url}
              alt={sourceImage.title}
              style={{ width: '100%', height: 'auto' }}
            />

            {/* Render Components (Bounding Boxes) */}
            {sourceImage.components
              .filter(comp => (comp.relevance_score || 0) >= threshold)
              .map((comp, i) => {
                const relevanceScore = comp.relevance_score || 0;
                const visualTier = getVisualTier(relevanceScore);

                return (
                  <div
                    key={i}
                    className="bounding-box"
                    title={`${comp.label} (${Math.round(relevanceScore * 100)}% relevant)`}
                    style={{
                      position: 'absolute',
                      left: `${comp.bbox_2d[0] / 10}%`,
                      top: `${comp.bbox_2d[1] / 10}%`,
                      width: `${(comp.bbox_2d[2] - comp.bbox_2d[0]) / 10}%`,
                      height: `${(comp.bbox_2d[3] - comp.bbox_2d[1]) / 10}%`,
                      border: `${visualTier.border_width}px solid ${visualTier.color}`,
                      opacity: visualTier.opacity,
                      backgroundColor: `${visualTier.color}20`,
                      boxSizing: 'border-box',
                      pointerEvents: 'none',
                      transition: 'opacity 0.2s ease, border-color 0.2s ease',
                    }}
                  >
                    <div
                      className="component-label"
                      style={{
                        position: 'absolute',
                        top: '-22px',
                        left: '0',
                        fontSize: visualTier.font_size,
                        fontWeight: visualTier.font_weight,
                        color: visualTier.color,
                        backgroundColor: 'rgba(255, 255, 255, 0.9)',
                        padding: '2px 6px',
                        borderRadius: '3px',
                        whiteSpace: 'nowrap',
                        maxWidth: '200px',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {comp.label}
                    </div>
                  </div>
                );
              })}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ComponentRenderer;
```

---

## CSS Styling

```css
.component-renderer {
  width: 100%;
}

.controls {
  padding: 16px;
  background-color: #f9fafb;
  border-radius: 8px;
  margin-bottom: 16px;
}

.controls label {
  display: block;
  font-weight: 600;
  margin-bottom: 8px;
  color: #374151;
}

.threshold-slider {
  width: 100%;
  margin-bottom: 12px;
}

.preset-buttons {
  display: flex;
  gap: 8px;
}

.preset-buttons button {
  flex: 1;
  padding: 8px 16px;
  background-color: #fff;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: background-color 0.2s, border-color 0.2s;
}

.preset-buttons button:hover {
  background-color: #f3f4f6;
  border-color: #9ca3af;
}

.preset-buttons button:active {
  background-color: #e5e7eb;
}

.source-images {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
}

.image-container {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.bounding-box {
  cursor: help;
}

.bounding-box:hover {
  opacity: 1 !important;
  z-index: 10;
}

.component-label {
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}
```

---

## Expected Results

### Query: "Tie down procedures for helicopters"

**Before (All components at full opacity):**
- 11 bounding boxes, all equally prominent
- Visual clutter, hard to focus

**After (4-tier gradual hierarchy):**

| Tier | Components | Visual Style |
|------|------------|--------------|
| **Tier 2 (Relevant, score 0.25)** | 4 tiedown diagrams | Opacity 0.9, blue border (3px), 12px font |
| **Tier 4 (Low, score 0.0)** | 7 knot diagrams | Opacity 0.3, gray border (1px), 10px font |

**Result:**
- Tiedown diagrams are prominent (blue, solid)
- Knot diagrams are de-emphasized (gray, faded) but still visible
- Clear visual hierarchy guides user attention
- No information loss

---

## Testing the Implementation

1. **Start the server:**
   ```bash
   python server.py
   ```

2. **Test the new endpoint:**
   ```bash
   curl http://localhost:8000/config/ux
   ```

3. **Send a test query:**
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Tie down procedures for helicopters"}'
   ```

4. **Verify component scores:**
   - Check that `source_images[].components[].relevance_score` is present
   - Verify scores match expected values (0.25 for tiedown components)

5. **Implement frontend rendering:**
   - Use either Vanilla JS or React example above
   - Verify 4-tier visual hierarchy appears correctly
   - Test threshold slider functionality

---

## Troubleshooting

### Components don't have `relevance_score`
- Ensure you're using the updated `chatbot.py` with substring containment
- Check that query is being passed to `_format_source_images()`
- Verify components list has >10 items (threshold for scoring)

### All components have score 0.0
- Check query tokenization (stopwords removed?)
- Verify component labels contain searchable terms
- Test with simpler queries like "valve" or "pump"

### Threshold slider doesn't update display
- Ensure you're re-rendering components on threshold change
- Check that `window.currentSourceImages` is set (Vanilla JS)
- Verify React state updates trigger re-render

---

## Advanced: Continuous Scaling (Optional)

For even smoother visual hierarchy, use continuous scaling instead of 4 tiers:

```javascript
function getVisualTierContinuous(relevanceScore) {
  // Map score 0-1 to opacity 0.3-1.0
  const opacity = 0.3 + (0.7 * relevanceScore);

  // Map score 0-1 to border width 1-3px
  const borderWidth = 1 + (2 * relevanceScore);

  // Interpolate color from gray to blue
  const color = interpolateColor('#9ca3af', '#2563eb', relevanceScore);

  // Map score 0-1 to font size 10-14px
  const fontSize = `${10 + (4 * relevanceScore)}px`;

  // Binary font weight at 0.5 threshold
  const fontWeight = relevanceScore > 0.5 ? '600' : '400';

  return { opacity, color, borderWidth, fontSize, fontWeight };
}

function interpolateColor(color1, color2, factor) {
  const c1 = hexToRgb(color1);
  const c2 = hexToRgb(color2);

  const r = Math.round(c1.r + (c2.r - c1.r) * factor);
  const g = Math.round(c1.g + (c2.g - c1.g) * factor);
  const b = Math.round(c1.b + (c2.b - c1.b) * factor);

  return `rgb(${r}, ${g}, ${b})`;
}

function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 0, g: 0, b: 0 };
}
```

This provides infinite gradations instead of 4 discrete tiers, but is slightly more complex to implement.
