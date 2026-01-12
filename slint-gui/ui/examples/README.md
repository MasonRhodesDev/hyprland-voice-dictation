# UI Style Examples

Example styles for the voice dictation overlay. Copy any style to `dictation.slint` to use it.

## Available Styles

### style1-default.slint
Full overlay with spectrum visualizer and transcription text.
- 380x90px rounded rectangle
- Spectrum bars + text display
- Best for: seeing real-time transcription

### style2-minimal.slint
Minimal horizontal pill with mirrored spectrum only.
- 200x40px pill shape
- Vertically mirrored spectrum bars (no text)
- Best for: unobtrusive visual feedback

## Customization

1. Copy the style you want:
   ```bash
   cp examples/style2-minimal.slint dictation.slint
   ```

2. The daemon will automatically reload when you save changes to `dictation.slint`.

3. Edit `dictation.slint` to customize colors, sizes, animations, etc.

## Creating Your Own Style

All styles must export a `Dictation` component with these properties:

```slint
export component Dictation inherits Window {
    in property <int> mode: 0;           // 0=hidden, 1=listening, 2=processing, 3=closing
    in property <[float]> spectrum;      // 8 frequency band values (0.0-1.0)
    in property <string> text;           // Transcription text
    in property <float> fade: 1.0;       // Overall opacity
    in property <float> closing-progress;// Collapse animation (0.0-1.0)
    in property <bool> pre-listening;    // True before audio starts

    background: transparent;
    // ... your UI here
}
```
