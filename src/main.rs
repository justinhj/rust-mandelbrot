use anyhow::{anyhow, Context, Result};
use image::ColorType;
use num::Complex;
use std::env;
use std::io::{stdin, stdout, Write};
use std::str::FromStr;
use termion::color;
use termion::event::{Event, Key};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;

struct Config {
    file_path: String,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
    num_threads: usize,
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

    let window = Rect {
        upper_left: config.upper_left,
        lower_right: config.lower_right,
    };

    let mut pixels = vec![0; config.bounds.0 * config.bounds.1];

    tui_loop(
        &config.file_path,
        config.num_threads,
        &mut pixels,
        config.bounds,
        &window,
    )?;

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

    let c = if complex_width > 0.0 {
        ((complex.re - window.upper_left.re) / complex_width * (width - 1) as f64).round() as i32
    } else {
        0
    };

    let r = if complex_height > 0.0 {
        ((window.upper_left.im - complex.im) / complex_height * (height - 1) as f64).round() as i32
    } else {
        0
    };

    (
        (c.clamp(0, width as i32 - 1) + 1) as u16,
        (r.clamp(0, height as i32 - 1) + 1) as u16,
    )
}

/// Draw the pixels of the Mandelbrot set as best we can in the terminal
/// https://www.unicode.org/charts/PDF/U2580.pdf
/// Window is the complex coordinates of the window into the Mandelbrot set
/// Selection is the current user selection for the next zoom
fn terminal_render(
    pixels: &[u8],
    bounds: (usize, usize),
    window: &Rect,
    selection: &Rect,
    terminal_size: (u16, u16),
    terminal_window_offset: (u16, u16),
) -> Result<()> {
    let mut stdout = std::io::BufWriter::new(std::io::stdout());
    let (cols, rows) = terminal_size;

    // Draw the pixels
    for c in 1..cols {
        let x_start = (c - 1) as usize * bounds.0 / cols as usize;
        let x_end = c as usize * bounds.0 / cols as usize;

        for r in 1..rows {
            let y_start = (r - 1) as usize * bounds.1 / rows as usize;
            let y_end = r as usize * bounds.1 / rows as usize;

            let mut sum: usize = 0;
            let mut count: usize = 0;
            for x in x_start..x_end {
                for y in y_start..y_end {
                    sum += pixels[x + y * bounds.0] as usize;
                    count += 1;
                }
            }

            let avg = if count > 0 { sum / count } else { 0 };
            let pixel = (avg * 24 / 256) as u8;

            write!(
                stdout,
                "{}{}\u{2588}",
                termion::cursor::Goto(c + terminal_window_offset.0, r + terminal_window_offset.1),
                termion::color::Fg(termion::color::AnsiValue::grayscale(pixel))
            )?;
        }
    }

    // Draw the zoom selection
    let (start_c, start_r) =
        complex_to_cursor_position(&selection.upper_left, window, (cols - 1, rows - 1));

    let (end_c, end_r) =
        complex_to_cursor_position(&selection.lower_right, window, (cols - 1, rows - 1));

    let selection_colour = color::AnsiValue::grayscale(20);

    // Top and bottom
    for c in start_c..=end_c {
        write!(
            stdout,
            "{}{}\u{2588}",
            termion::cursor::Goto(c + terminal_window_offset.0, start_r + terminal_window_offset.1),
            termion::color::Fg(selection_colour)
        )?;
        write!(
            stdout,
            "{}{}\u{2588}",
            termion::cursor::Goto(c + terminal_window_offset.0, end_r + terminal_window_offset.1),
            termion::color::Fg(selection_colour)
        )?;
    }

    // Sides
    for r in start_r..=end_r {
        write!(
            stdout,
            "{}{}\u{2588}",
            termion::cursor::Goto(start_c + terminal_window_offset.0, r + terminal_window_offset.1),
            termion::color::Fg(selection_colour)
        )?;
        write!(
            stdout,
            "{}{}\u{2588}",
            termion::cursor::Goto(end_c + terminal_window_offset.0, r + terminal_window_offset.1),
            termion::color::Fg(selection_colour)
        )?;
    }
    stdout.flush()?;
    Ok(())
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

/// Event loop and renderer
/// This enables the user to write the image, move and zoom the selection window 
/// and so on.
fn tui_loop(
    file_path: &str,
    num_threads: usize,
    pixels: &mut [u8],
    bounds: (usize, usize),
    initial_window: &Rect,
) -> Result<()> {
    let mut numbered_file_path = increment_numbered_filename(file_path);
    let zoom = 0.5f64;
    let stdin = stdin();
    let mut stdout = MouseTerminal::from(
        stdout()
            .into_raw_mode()
            .context("failed to set raw mode")?,
    );

    let mut ts = termion::terminal_size().context("failed to get terminal size")?;
    let mut window = initial_window.clone();

    let selection = selection_from_window(&window, zoom);

    // Temp for debugging in IntelliJ the terminal has zero size
    if ts.0 == 0 {
        ts = (120, 80);
    }

    write!(
        stdout,
        "{}{}Esc to exit - wasd to move selection - Enter to write current image file and z to zoom",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .context("failed to write to stdout")?;

    // Render the image to the pizel array
    parallel_render(
        pixels,
        bounds,
        window.upper_left,
        window.lower_right,
        num_threads,
    );
    terminal_render(
        pixels,
        bounds,
        &window,
        &selection,
        (ts.0, ts.1 - 1),
        (0, 1),
    )?;
    stdout.flush().context("failed to flush stdout")?;

    let mut moving_selection = selection.clone();

    let terminal_bounds = (ts.0, ts.1 - 1);

    for c in stdin.events() {
        let evt = c.context("failed to read event")?;
        if let Event::Key(key) = evt {
            match key {
                Key::Esc => {
                    write!(stdout, "{}", termion::clear::All).context("failed to clear screen")?;
                    break;
                }
                Key::Char('a') => {
                    moving_selection = moving_selection.move_by(&window, terminal_bounds, -1, 0);
                }
                Key::Char('d') => {
                    moving_selection = moving_selection.move_by(&window, terminal_bounds, 1, 0);
                }
                Key::Char('w') => {
                    moving_selection = moving_selection.move_by(&window, terminal_bounds, 0, 1);
                }
                Key::Char('s') => {
                    moving_selection = moving_selection.move_by(&window, terminal_bounds, 0, -1);
                }

                Key::Char('\n') => {
                    write_image(&numbered_file_path, pixels, bounds)
                        .context("error writing PNG file")?;
                    numbered_file_path = increment_numbered_filename(&numbered_file_path);
                }
                Key::Char('z') => {
                    window.clone_from(&moving_selection);
                    moving_selection = selection_from_window(&window, zoom);
                    parallel_render(
                        pixels,
                        bounds,
                        window.upper_left,
                        window.lower_right,
                        num_threads,
                    );
                }
                _ => {}
            }
        }
        terminal_render(
            pixels,
            bounds,
            &window,
            &moving_selection,
            (ts.0, ts.1 - 1),
            (0, 1),
        )?;
        stdout.flush().context("failed to flush stdout")?;
    }
    Ok(())
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

fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<()> {
    image::save_buffer(
        filename,
        pixels,
        bounds.0 as u32,
        bounds.1 as u32,
        ColorType::L8,
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
        // (25-0)/100 * 199 = 49.75 -> 50. 50 + 1 = 51.
        // (100-75)/100 * 199 = 49.75 -> 50. 50 + 1 = 51.
        assert_eq!((c, r), (51, 51));
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
    fn test_selection_move_preserves_size() {
        let window = Rect {
            upper_left: Complex { re: -2.0, im: 1.0 },
            lower_right: Complex { re: 1.0, im: -1.0 },
        };

        let selection = Rect {
            upper_left: Complex { re: -1.0, im: 0.5 },
            lower_right: Complex { re: 0.5, im: -0.5 },
        };

        let terminal_size = (100, 100);

        // Original width and height
        let original_width = (selection.lower_right.re - selection.upper_left.re).abs();
        let original_height = (selection.upper_left.im - selection.lower_right.im).abs();

        // Move right towards the edge
        // complex_x_per_char = 3.0 / 100 = 0.03
        // To reach the edge (re=1.0) from re=0.5, we need 0.5 / 0.03 = 16.66 chars.
        // Let's move 20 chars.
        let moved = selection.move_by(&window, terminal_size, 20, 0);

        let new_width = (moved.lower_right.re - moved.upper_left.re).abs();
        let new_height = (moved.upper_left.im - moved.lower_right.im).abs();

        assert!(
            (new_width - original_width).abs() < 1e-10,
            "Width changed from {} to {}",
            original_width,
            new_width
        );
        assert!(
            (new_height - original_height).abs() < 1e-10,
            "Height changed from {} to {}",
            original_height,
            new_height
        );
        
        // Ensure it stopped at the edge
        assert!(moved.lower_right.re <= 1.0 + 1e-10);
        assert!(moved.upper_left.re >= -2.0 - 1e-10);

        // Move left towards the edge
        let moved = selection.move_by(&window, terminal_size, -40, 0);
        let new_width = (moved.lower_right.re - moved.upper_left.re).abs();
        assert!((new_width - original_width).abs() < 1e-10);
        assert!(moved.upper_left.re >= -2.0 - 1e-10);

        // Move up towards the edge
        let moved = selection.move_by(&window, terminal_size, 0, 20);
        let new_height = (moved.upper_left.im - moved.lower_right.im).abs();
        assert!((new_height - original_height).abs() < 1e-10);
        assert!(moved.upper_left.im <= 1.0 + 1e-10);

        // Move down towards the edge
        let moved = selection.move_by(&window, terminal_size, 0, -20);
        let new_height = (moved.upper_left.im - moved.lower_right.im).abs();
        assert!((new_height - original_height).abs() < 1e-10);
        assert!(moved.lower_right.im >= -1.0 - 1e-10);
    }
}
