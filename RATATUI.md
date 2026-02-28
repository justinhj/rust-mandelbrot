# Ratatui Migration Plan

This document outlines the plan to migrate `rust-mandelbrot` from a manual `termion` renderer to the `ratatui` library. This migration will resolve the flickering issue (#1) and provide a robust framework for improving TUI fidelity (#4) and adding configuration menus (#6).

## 1. Objectives
*   **Eliminate Flickering:** Use Ratatui's internal double-buffering and diffing to only update changed cells.
*   **Improved Layout:** Use Ratatui's layout system to manage the instruction bar and the Mandelbrot visualization area.
*   **Cross-Platform Support:** Transition to the `crossterm` backend, which is the modern standard for Ratatui and provides better Windows support.
*   **Foundation for High Fidelity:** Prepare for half-block rendering (`\u{2580}`) to double vertical resolution.

## 2. Dependency Changes
Add the following to `Cargo.toml`:
```toml
[dependencies]
ratatui = { version = "0.29", features = ["crossterm"] }
crossterm = { version = "0.28", features = ["event"] }
```
Remove `termion` once the migration is complete.

## 3. Architectural Changes

### 3.1 App State Structure
Consolidate the application state into a single `App` struct:
```rust
struct App {
    pixels: Vec<u8>,
    bounds: (usize, usize),
    window: Rect,
    selection: Rect,
    num_threads: usize,
    should_quit: bool,
}
```

### 3.2 Rendering Logic
Refactor `terminal_render` into a function that renders to a `ratatui::Frame`:
*   Map Mandelbrot pixels to `ratatui::widgets::canvas` or directly to `ratatui::buffer::Buffer`.
*   Use `ratatui::style::Color::Rgb(r, g, b)` for truecolor support.
*   Draw the selection box as a `ratatui::widgets::Block` or manual buffer manipulation.

### 3.3 Event Loop
Replace the current `stdin.events()` loop with a `ratatui` terminal loop:
1.  Initialize `crossterm` raw mode and alternate screen.
2.  In a loop:
    *   `terminal.draw(|f| ui(f, &app))?`
    *   Handle `crossterm::event::read()` for keyboard inputs.
    *   Update `App` state based on inputs.

## 4. Implementation Steps

### Phase 1: Infrastructure
1.  Update `Cargo.toml` with `ratatui` and `crossterm`.
2.  Create `App` struct and move existing configuration logic there.
3.  Implement terminal setup/teardown helpers (raw mode, alternate screen, mouse support).

### Phase 2: Core Rendering
1.  Implement the `ui` function to define the layout (Header + Main Area).
2.  Convert Mandelbrot pixel mapping to write into Ratatui's `Buffer`.
3.  Optimize the inner loop: avoid redundant color calculations by using the diffing engine's strengths.

### Phase 3: Input & State
1.  Migrate keyboard handling from `termion::event::Key` to `crossterm::event::KeyCode`.
2.  Ensure `parallel_render` is called only when the Mandelbrot window changes (zooming).
3.  Verify that moving the selection box only triggers a UI redraw, not a full Mandelbrot recalculation.

### Phase 4: Cleanup
1.  Remove `termion` dependency.
2.  Validate truecolor support detection using `crossterm` or environment variables.

## 5. Future Enhancements enabled by Ratatui
*   **Split View:** Show the current render and a "preview" of the zoom area simultaneously.
*   **Interactive Menu:** Use Ratatui's `List` or `Table` widgets for the configuration menu (#6).
*   **Progress Bars:** Show calculation progress for high-resolution renders using `ratatui::widgets::Gauge`.
