use anyhow::{anyhow, Context, Result};
use image::ColorType;
use num::Complex;
use std::env;
use std::io;
use std::str::FromStr;

use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::Paragraph,
    Frame, Terminal,
};
use crossterm::{
    event::{self, Event as CEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

struct Config {
    file_path: String,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
    num_threads: usize,
}

struct App {
    pixels: Vec<u8>,
    bounds: (usize, usize),
    window: Rect,
    moving_selection: Rect,
    num_threads: usize,
    numbered_file_path: String,
}

impl App {
    fn new(config: Config) -> Self {
        let window = Rect {
            upper_left: config.upper_left,
            lower_right: config.lower_right,
        };
        let selection = selection_from_window(&window, 0.5);
        let numbered_file_path = increment_numbered_filename(&config.file_path);

        App {
            pixels: vec![0; config.bounds.0 * config.bounds.1],
            bounds: config.bounds,
            window,
            moving_selection: selection,
            num_threads: config.num_threads,
            numbered_file_path,
        }
    }

    fn update_mandelbrot(&mut self) {
        parallel_render(
            &mut self.pixels,
            self.bounds,
            self.window.upper_left,
            self.window.lower_right,
            self.num_threads,
        );
    }

    fn move_selection(&mut self, cols: i32, rows: i32, terminal_size: (u16, u16)) {
        self.moving_selection =
            self.moving_selection
                .move_by(&self.window, terminal_size, cols, rows);
    }

    fn zoom(&mut self) {
        self.window.clone_from(&self.moving_selection);
        self.moving_selection = selection_from_window(&self.window, 0.5);
        self.update_mandelbrot();
    }

    fn save_image(&mut self) -> Result<()> {
        write_image(&self.numbered_file_path, &self.pixels, self.bounds)?;
        self.numbered_file_path = increment_numbered_filename(&self.numbered_file_path);
        Ok(())
    }
}

fn parse_args() -> Result<Config> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        return Err(anyhow!(
            "Usage: {} FILE PIXELS UPPERLEFT UPPERRIGHT [THREADS]\n\
             Example: {} ./images/mandel.png 2560x1440 -1.20,1.0 1.20,-1.0 4",
            args[0],
            args[0]
        ));
    }

    let file_path = args[1].clone();
    let bounds = parse_pair(&args[2], 'x').context("error parsing image dimensions")?;
    let upper_left = parse_complex(&args[3]).context("error parsing upper left corner point")?;
    let lower_right = parse_complex(&args[4]).context("error parsing lower right corner point")?;

    let num_threads = if args.len() == 6 {
        args[5].parse().context("error parsing thread count")?
    } else {
        num_cpus::get()
    };

    if num_threads == 0 {
        return Err(anyhow!("thread count must be greater than zero"));
    }

    Ok(Config {
        file_path,
        bounds,
        upper_left,
        lower_right,
        num_threads,
    })
}

fn main() -> Result<()> {
    let config = match parse_args() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    let mut app = App::new(config);
    app.update_mandelbrot();

    // setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, event::EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // run app
    let res = run_app(&mut terminal, &mut app);

    // restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        event::DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct Rect {
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
}

impl Rect {
    /// Move the rectangle by a number of characters within a bounding window
    fn move_by(&self, window: &Rect, terminal_size: (u16, u16), delta_cols: i32, delta_rows: i32) -> Rect {
        let complex_x_per_char = (window.lower_right.re - window.upper_left.re).abs() / terminal_size.0 as f64;
        let complex_y_per_char = (window.upper_left.im - window.lower_right.im).abs() / terminal_size.1 as f64;

        let mut dx = delta_cols as f64 * complex_x_per_char;
        let mut dy = delta_rows as f64 * complex_y_per_char;

        let min_re = f64::min(window.upper_left.re, window.lower_right.re);
        let max_re = f64::max(window.upper_left.re, window.lower_right.re);
        let min_im = f64::min(window.upper_left.im, window.lower_right.im);
        let max_im = f64::max(window.upper_left.im, window.lower_right.im);

        let sel_min_re = f64::min(self.upper_left.re, self.lower_right.re);
        let sel_max_re = f64::max(self.upper_left.re, self.lower_right.re);
        let sel_min_im = f64::min(self.upper_left.im, self.lower_right.im);
        let sel_max_im = f64::max(self.upper_left.im, self.lower_right.im);

        if dx > 0.0 {
            dx = dx.min(max_re - sel_max_re);
        } else if dx < 0.0 {
            dx = dx.max(min_re - sel_min_re);
        }

        if dy > 0.0 {
            dy = dy.min(max_im - sel_max_im);
        } else if dy < 0.0 {
            dy = dy.max(min_im - sel_min_im);
        }

        Rect {
            upper_left: Complex {
                re: self.upper_left.re + dx,
                im: self.upper_left.im + dy,
            },
            lower_right: Complex {
                re: self.lower_right.re + dx,
                im: self.lower_right.im + dy,
            },
        }
    }
}

/// Given bounds in cursor space and a Complex number return the cursor position (col,row)
fn complex_to_cursor_position(
    complex: &Complex<f64>,
    window: &Rect,
    terminal_bounds: (u16, u16),
) -> (u16, u16) {
    let (width, height) = terminal_bounds;

    let complex_width = (window.lower_right.re - window.upper_left.re).abs();
    let complex_height = (window.upper_left.im - window.lower_right.im).abs();

    let c = if complex_width > 0.0 && width > 1 {
        ((complex.re - window.upper_left.re) / complex_width * (width as f64 - 1.0)).round() as i32
    } else {
        0
    };

    let r = if complex_height > 0.0 && height > 1 {
        ((window.upper_left.im - complex.im) / complex_height * (height as f64 - 1.0)).round() as i32
    } else {
        0
    };

    (
        c.clamp(0, width as i32 - 1) as u16,
        r.clamp(0, height as i32 - 1) as u16,
    )
}

fn selection_from_window(window: &Rect, _zoom: f64) -> Rect {
    // TODO use zoom
    let quarter_width = num::abs(window.lower_right.re - window.upper_left.re) / 4.0;
    let quarter_height = num::abs(window.upper_left.im - window.lower_right.im) / 4.0;
    Rect {
        upper_left: Complex {
            re: window.upper_left.re + quarter_width,
            im: window.upper_left.im - quarter_height,
        },
        lower_right: Complex {
            re: window.lower_right.re - quarter_width,
            im: window.lower_right.im + quarter_height,
        },
    }
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if event::poll(std::time::Duration::from_millis(16))? {
            if let CEvent::Key(key) = event::read()? {
                let size = terminal.size()?;
                let mandelbrot_area = (size.width, size.height.saturating_sub(1));

                match key.code {
                    KeyCode::Esc => return Ok(()),
                    KeyCode::Char('a') => app.move_selection(-1, 0, mandelbrot_area),
                    KeyCode::Char('d') => app.move_selection(1, 0, mandelbrot_area),
                    KeyCode::Char('w') => app.move_selection(0, 1, mandelbrot_area),
                    KeyCode::Char('s') => app.move_selection(0, -1, mandelbrot_area),
                    KeyCode::Char('z') => app.zoom(),
                    KeyCode::Enter => {
                        app.save_image().expect("failed to save image");
                    }
                    _ => {}
                }
            }
        }
    }
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(f.area());

    let help_text = "Esc to exit - wasd to move selection - Enter to write current image file and z to zoom";
    let help_paragraph = Paragraph::new(help_text);
    f.render_widget(help_paragraph, chunks[0]);

    render_mandelbrot(f, app, chunks[1]);
}

fn render_mandelbrot(f: &mut Frame, app: &App, area: ratatui::layout::Rect) {
    let (cols, rows) = (area.width, area.height);
    let buf = f.buffer_mut();

    // Draw pixels
    for r in 0..rows {
        let r_idx = area.y + r;
        let y_start = r as usize * app.bounds.1 / rows as usize;
        let y_end = (r + 1) as usize * app.bounds.1 / rows as usize;

        for c in 0..cols {
            let c_idx = area.x + c;
            let x_start = c as usize * app.bounds.0 / cols as usize;
            let x_end = (c + 1) as usize * app.bounds.0 / cols as usize;

            let mut sum: usize = 0;
            let mut count: usize = 0;
            for x in x_start..x_end {
                for y in y_start..y_end {
                    sum += app.pixels[x + y * app.bounds.0] as usize;
                    count += 1;
                }
            }

            let avg = if count > 0 { (sum / count) as u8 } else { 0 };
            let (r, g, b) = palette(avg);

            buf[(c_idx, r_idx)]
                .set_char('\u{2588}')
                .set_style(Style::default().fg(Color::Rgb(r, g, b)));
        }
    }

    // Draw selection
    let (start_c, start_r) =
        complex_to_cursor_position(&app.moving_selection.upper_left, &app.window, (cols, rows));
    let (end_c, end_r) =
        complex_to_cursor_position(&app.moving_selection.lower_right, &app.window, (cols, rows));

    let selection_style = Style::default().fg(Color::Indexed(20));

    // Top & Bottom
    for c in start_c..=end_c {
        if c < cols {
            buf[(area.x + c, area.y + start_r)]
                .set_style(selection_style);
            buf[(area.x + c, area.y + end_r)]
                .set_style(selection_style);
        }
    }
    // Sides
    for r in start_r..=end_r {
        if r < rows {
            buf[(area.x + start_c, area.y + r)]
                .set_style(selection_style);
            buf[(area.x + end_c, area.y + r)]
                .set_style(selection_style);
        }
    }
}

/// Given a pixel array, bounds and window co-ordinates and number of threads to use
/// render to the pixel array
fn parallel_render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
    num_threads: usize,
) {
    let rows_per_band = bounds.1 / num_threads + 1;

    let bands: Vec<&mut [u8]> = pixels.chunks_mut(rows_per_band * bounds.0).collect();
    crossbeam::scope(|spawner| {
        for (i, band) in bands.into_iter().enumerate() {
            let top = rows_per_band * i;
            let height = band.len() / bounds.0;
            let band_bounds = (bounds.0, height);
            let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
            let band_lower_right =
                pixel_to_point(bounds, (bounds.0, top + height), upper_left, lower_right);
            spawner.spawn(move |_| {
                render(band, band_bounds, band_upper_left, band_lower_right);
            });
        }
    })
    .unwrap();
}

/// Try to determine if `c` is in the Mandelbrot set, using at most `limit`
/// iterations to decide.
///
/// If `c` is not a member, return `Some(i)`, where `i` is the number of
/// iterations it took for `c` to leave the circle of radius 2 centered on the
/// origin. If `c` seems to be a member (more precisely, if we reached the
/// iteration limit without being able to prove that `c` is not a member),
/// return `None`.
fn escape_time(c: Complex<f64>, limit: usize) -> Option<usize> {
    let mut z = Complex { re: 0.0, im: 0.0 };

    for i in 0..limit {
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
        z = z * z + c;
    }
    None
}

// eg 1000 pixels to 100 chars
// pixels per char = pixel w / char w ... round down

/// Calculate the Complex number that represents the pixel within an image of the defined bounds
fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    Complex {
        re: upper_left.re + pixel.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64,
    }
}

fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    match s.find(separator) {
        None => None,
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Some((l, r)),
            _ => None,
        },
    }
}

// Render the Mandelbrot set into the pixel array
fn render(
    pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for col in 0..bounds.0 {
            let point = pixel_to_point(bounds, (col, row), upper_left, lower_right);
            pixels[row * bounds.0 + col] = match escape_time(point, 255) {
                None => 0,
                Some(count) => 255 - count as u8,
            }
        }
    }
}

fn parse_complex(s: &str) -> Option<Complex<f64>> {
    let parsed: Option<(f64, f64)> = parse_pair(s, ',');
    parsed.map(|(re, im)| Complex::new(re, im))
}

fn palette(t: u8) -> (u8, u8, u8) {
    if t == 0 {
        return (0, 0, 0);
    }
    let t = t as f32 / 255.0;
    let r = (0.5 + 0.5 * (6.28 * (1.0 * t + 0.0)).cos()) * 255.0;
    let g = (0.5 + 0.5 * (6.28 * (1.0 * t + 0.33)).cos()) * 255.0;
    let b = (0.5 + 0.5 * (6.28 * (1.0 * t + 0.67)).cos()) * 255.0;
    (r as u8, g as u8, b as u8)
}

fn has_truecolor() -> bool {
    match std::env::var("COLORTERM") {
        Ok(val) => val == "truecolor" || val == "24bit",
        Err(_) => false,
    }
}

fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<()> {
    let mut rgb_pixels = Vec::with_capacity(pixels.len() * 3);
    for &p in pixels {
        let (r, g, b) = palette(p);
        rgb_pixels.push(r);
        rgb_pixels.push(g);
        rgb_pixels.push(b);
    }
    image::save_buffer(
        filename,
        &rgb_pixels,
        bounds.0 as u32,
        bounds.1 as u32,
        ColorType::Rgb8,
    )
    .context("failed to save image")?;
    Ok(())
}

fn increment_numbered_filename(file_path: &str) -> String {
    let path = std::path::Path::new(file_path);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    let first_non_numeric = stem.rfind(|c: char| !c.is_numeric());

    let (prefix, num_str) = match first_non_numeric {
        Some(idx) => stem.split_at(idx + 1),
        None => ("", stem),
    };

    let next_num = num_str.parse::<u64>().map(|n| n + 1).unwrap_or(1);
    let next_stem = format!("{}{}", prefix, next_num);

    let mut next_path = path.to_path_buf();
    next_path.set_file_name(next_stem);
    if !extension.is_empty() {
        next_path.set_extension(extension);
    }

    next_path.to_str().unwrap_or(file_path).to_string()
}

#[test]
fn test_parse_pair() {
    assert_eq!(parse_pair::<i32>("", ','), None);
    assert_eq!(parse_pair::<i32>("10,", ','), None);
    assert_eq!(parse_pair::<i32>(",10", ','), None);
    assert_eq!(parse_pair::<i32>("10,20", ','), Some((10, 20)));
    assert_eq!(parse_pair::<i32>("10,20xy", ','), None);
    assert_eq!(parse_pair::<f64>("0.5x", 'x'), None);
    assert_eq!(parse_pair::<f64>("0.5x1.5", 'x'), Some((0.5, 1.5)));
}

#[test]
fn test_parse_complex() {
    assert_eq!(
        parse_complex(&"1.25,-0.0625".to_string()),
        Some(Complex {
            re: 1.25,
            im: -0.0625
        })
    );
    assert_eq!(parse_complex(&",-0.0625".to_string()), None);
}

#[test]
fn test_pixel_to_point() {
    assert_eq!(
        pixel_to_point(
            (100, 200),
            (25, 175),
            Complex { re: -1.0, im: 1.0 },
            Complex { re: 1.0, im: -1.0 }
        ),
        Complex {
            re: -0.5,
            im: -0.75
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // In tests I compare Rect but since you can't easily compare f64, I use the float_cmp crate
    // and implement PartialEq and Eq...
    impl PartialEq for Rect {
        fn eq(&self, other: &Self) -> bool {
            float_cmp::approx_eq!(f64, self.upper_left.re, other.upper_left.re, ulps = 3)
                && float_cmp::approx_eq!(f64, self.upper_left.im, other.upper_left.im, ulps = 3)
                && float_cmp::approx_eq!(f64, self.lower_right.re, other.lower_right.re, ulps = 3)
                && float_cmp::approx_eq!(f64, self.lower_right.im, other.lower_right.im, ulps = 3)
        }
    }
    impl Eq for Rect {}

    #[test]
    fn test_complex_to_cursor_position() {
        let window = Rect {
            upper_left: Complex { re: 0.0, im: 100.0 },
            lower_right: Complex { re: 100.0, im: 0.0 },
        };
        let selection = selection_from_window(&window, 1.0);
        assert_eq!(
            selection,
            Rect {
                upper_left: Complex { re: 25.0, im: 75.0 },
                lower_right: Complex { re: 75.0, im: 25.0 }
            }
        );

        let terminal_size = (200, 200);
        let (c, r) = complex_to_cursor_position(&selection.upper_left, &window, terminal_size);
        // (25-0)/100 * 199 = 49.75 -> 50.
        // (100-75)/100 * 199 = 49.75 -> 50.
        assert_eq!((c, r), (50, 50));
    }

    #[test]
    fn test_numbered_path() {
        let sample_path: String = "/tmp/mandelbrot.png".to_string();

        let numbered_1 = increment_numbered_filename(&sample_path);
        assert_eq!(numbered_1, "/tmp/mandelbrot1.png");

        let numbered_2 = increment_numbered_filename(&numbered_1);
        assert_eq!(numbered_2, "/tmp/mandelbrot2.png");

        let numbered_20: String = (1..=200u64).fold(sample_path.to_string(), |acc, _| {
            increment_numbered_filename(&acc)
        });
        assert_eq!(numbered_20, "/tmp/mandelbrot200.png");
    }

    #[test]
    fn test_palette() {
        let (r, g, b) = palette(0);
        assert_eq!((r, g, b), (0, 0, 0));
        
        let (r, g, b) = palette(128);
        // Just ensure it's not all zeros for a non-zero input
        assert!(r > 0 || g > 0 || b > 0);
    }

    #[test]
    fn test_write_image() -> Result<()> {
        let pixels = vec![0, 64, 128, 255];
        let bounds = (2, 2);
        let temp_file = "test_output.png";
        write_image(temp_file, &pixels, bounds)?;
        assert!(std::path::Path::new(temp_file).exists());
        std::fs::remove_file(temp_file)?;
        Ok(())
    }

    #[test]
    fn test_has_truecolor() {
        let original = std::env::var("COLORTERM");
        
        unsafe {
            std::env::set_var("COLORTERM", "truecolor");
            assert!(has_truecolor());
            std::env::set_var("COLORTERM", "24bit");
            assert!(has_truecolor());
            std::env::set_var("COLORTERM", "somethingelse");
            assert!(!has_truecolor());
            std::env::remove_var("COLORTERM");
            assert!(!has_truecolor());

            if let Ok(val) = original {
                std::env::set_var("COLORTERM", val);
            }
        }
    }
}
