use image::png::PNGEncoder;
use image::ColorType;
use num::Complex;
// use std::cmp;
use std::env;
use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::str::FromStr;
use termion::event::{Event, Key}; // , MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;
use termion::color;

#[derive(Debug,Clone)]
struct Rect {
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
}
impl PartialEq for Rect {
    fn eq(&self, other: &Self) -> bool {
        float_cmp::approx_eq!(f64, self.upper_left.re, other.upper_left.re, ulps = 3)
            && float_cmp::approx_eq!(f64, self.upper_left.im, other.upper_left.im, ulps = 3)
            && float_cmp::approx_eq!(f64, self.lower_right.re, other.lower_right.re, ulps = 3)
            && float_cmp::approx_eq!(f64, self.lower_right.im, other.lower_right.im, ulps = 3)
    }
}
impl Eq for Rect {}

fn main() {
    let args = env::args().collect::<Vec<String>>();

    let num_cpus = num_cpus::get();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} FILE PIXELS UPPERLEFT UPPERRIGHT [THREADS]",
            args[0]
        );
        eprintln!(
            "Example: {} mandel.png 2560x1440 -1.20,1.0 1.20,-1.0 4",
            args[0]
        );
        std::process::exit(1);
    }

    let bounds = parse_pair(&args[2], 'x').expect("error parsing image dimensions");
    let upper_left = parse_complex(&args[3]).expect("error parsing upper left corner point");
    let lower_right = parse_complex(&args[4]).expect("error parsing lower right corner point");

    let window = Rect {
        upper_left,
        lower_right,
    };

    let mut pixels = vec![0; bounds.0 * bounds.1];

    // Handle thread input
    let num_threads: usize = if args.len() == 6 {
        args[5].parse().unwrap()
    } else {
        num_cpus
    };

    assert!(num_threads > 0);

    parallel_render(&mut pixels, bounds, upper_left, lower_right, num_threads);

    tui_loop(&mut pixels, bounds, &window);

    write_image(&args[1], &pixels, bounds).expect("error writing PNG file");
}

// What is the width and height of a cursor

// Given a selection within a larger window and the cursor bounds
// return a new selection moved to the left the set number of cursor
// positions.
fn move_selection_left(
    selection: &Rect,
    window: &Rect,
    terminal_bounds: (u16, u16),
    count: u16,
) -> Rect {
    let complex_x_per_char =
        num::abs(window.lower_right.re - window.upper_left.re) / terminal_bounds.0 as f64;

    let upper_left_re = f64::max(selection.upper_left.re - (count as f64 * complex_x_per_char), window.upper_left.re); 
    let lower_right_re = f64::max(selection.lower_right.re - (count as f64 * complex_x_per_char), window.upper_left.re); 

    Rect {
        upper_left: Complex { re: upper_left_re, im: selection.upper_left.im },
        lower_right: Complex { re: lower_right_re, im: selection.lower_right.im },
    }
}

fn move_selection_right(
    selection: &Rect,
    window: &Rect,
    terminal_bounds: (u16, u16),
    count: u16,
) -> Rect {
    let complex_x_per_char =
        num::abs(window.lower_right.re - window.upper_left.re) / terminal_bounds.0 as f64;
    let upper_left_re = f64::min(selection.upper_left.re + (count as f64 * complex_x_per_char), window.lower_right.re); 
    let lower_right_re = f64::min(selection.lower_right.re + (count as f64 * complex_x_per_char), window.lower_right.re); 

    Rect {
        upper_left: Complex { re: upper_left_re, im: selection.upper_left.im },
        lower_right: Complex { re: lower_right_re, im: selection.lower_right.im },
    }
}

fn move_selection_down(
    selection: &Rect,
    window: &Rect,
    terminal_bounds: (u16, u16),
    count: u16,
) -> Rect {
    let complex_y_per_char =
        num::abs(window.upper_left.im - window.lower_right.im) / terminal_bounds.1 as f64;

    let upper_left_im = f64::min(selection.upper_left.im - (count as f64 * complex_y_per_char), window.upper_left.im); 
    let lower_right_im = f64::min(selection.lower_right.im - (count as f64 * complex_y_per_char), window.upper_left.im); 

    Rect {
        upper_left: Complex { re: selection.upper_left.re, im: upper_left_im},
        lower_right: Complex { re: selection.lower_right.re, im: lower_right_im},
    }
}

fn move_selection_up(
    selection: &Rect,
    window: &Rect,
    terminal_bounds: (u16, u16),
    count: u16,
) -> Rect {
    let complex_y_per_char =
        num::abs(window.upper_left.im - window.lower_right.im) / terminal_bounds.1 as f64;

    let upper_left_im = f64::max(selection.upper_left.im + (count as f64 * complex_y_per_char), window.lower_right.im); 
    let lower_right_im = f64::max(selection.lower_right.im + (count as f64 * complex_y_per_char), window.lower_right.im); 

    Rect {
        upper_left: Complex { re: selection.upper_left.re, im: upper_left_im},
        lower_right: Complex { re: selection.lower_right.re, im: lower_right_im},
    }
}

/// Given bounds in cursor space and a Complex number return the cursor position (col,row)
fn complex_to_cursor_position(
    complex: &Complex<f64>,
    window: &Rect,
    terminal_bounds: (u16, u16),
) -> (u16, u16) {
    // Calculate how much complex number is represented by one cursor position for rows and columns
    let complex_x_per_char =
        num::abs(window.lower_right.re - window.upper_left.re) / terminal_bounds.0 as f64;
    let complex_y_per_char =
        num::abs(window.upper_left.im - window.lower_right.im) / terminal_bounds.1 as f64;

    let c: f64 = num::abs(complex.re - window.upper_left.re) / complex_x_per_char;
    let r: f64 = num::abs(window.lower_right.im - complex.im) / complex_y_per_char;

    (c as u16, terminal_bounds.1 - (r as u16))
}

/// Draw the pixels of the Mandelbrot set as best we can in the terminal
/// https://www.unicode.org/charts/PDF/U2580.pdf
/// Window is the complex coordinates of the window into the Mandelbrot set
/// Selection is the current user selection for the next zoom
fn terminal_render(
    pixels: &Vec<u8>,
    bounds: (usize, usize),
    window: &Rect,
    selection: &Rect,
    terminal_size: (u16, u16),
    terminal_window_offset: (u16, u16),
) {
    let (cols, rows) = terminal_size;

    // Draw the pixels
    for c in 1..cols {
        for r in 1..rows {
            let pixel = pixel_to_char_grayscale((c, r), (cols, rows), pixels, bounds);

            println!(
                "{}{}\u{2588}",
                termion::cursor::Goto(c + terminal_window_offset.0, r + terminal_window_offset.1),
                termion::color::Fg(termion::color::AnsiValue::grayscale(pixel))
            );
        }
    }

    // Draw the zoom selection
    let (start_c, start_r) =
        complex_to_cursor_position(&selection.upper_left, window, terminal_size);

    let (end_c, end_r) = complex_to_cursor_position(&selection.lower_right, window, terminal_size);

    let selection_colour = color::AnsiValue::grayscale(20);

    // Top and bottom
    for c in start_c..end_c {
        println!(
            "{}{}\u{2588}",
            termion::cursor::Goto(c, end_r),
            termion::color::Fg(selection_colour)
        );
        println!(
            "{}{}\u{2588}",
            termion::cursor::Goto(c, start_r),
            termion::color::Fg(selection_colour)
        );
    }

    // Sides
    for r in start_r..end_r {
        println!(
            "{}{}\u{2588}",
            termion::cursor::Goto(start_c, r),
            termion::color::Fg(selection_colour)
        );
        println!(
            "{}{}\u{2588}",
            termion::cursor::Goto(end_c, r),
            termion::color::Fg(selection_colour)
        );
    }
}

fn selection_from_window(window: &Rect, _zoom: f64) -> Rect {
    // TODO use zoom
    let quarter_width = num::abs(window.lower_right.re - window.upper_left.re) / 4.0;
    let quarter_height = num::abs(window.upper_left.im - window.lower_right.im) / 4.0;
    return Rect {
        upper_left: Complex {
            re: window.upper_left.re + quarter_width,
            im: window.upper_left.im - quarter_height,
        },
        lower_right: Complex {
            re: window.lower_right.re - quarter_width,
            im: window.lower_right.im + quarter_height,
        },
    };
}

/// Render to the terminal
fn tui_loop(pixels: &mut Vec<u8>, bounds: (usize, usize), window: &Rect) {
    let zoom = 0.5f64;
    let selection = selection_from_window(&window, zoom);

    let stdin = stdin();
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());

    let mut ts = termion::terminal_size().unwrap();

    // Temp for debugging in IntelliJ the terminal has zero size
    if ts.0 == 0 {
        ts = (120, 80);
    }

    write!(
        stdout,
        "{}{}Esc to exit - wasd to move selection - z to zoom - space to write image",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();

    terminal_render(
        &pixels,
        bounds,
        &window,
        &selection,
        (ts.0, ts.1 - 1),
        (0, 1),
    );
    stdout.flush().unwrap();

    let mut moving_selection = selection.clone();

    for c in stdin.events() {
        let evt = c.unwrap();
        match evt {
            Event::Key(key) => match key {
                Key::Esc => {
                    println!("{}", termion::clear::All);
                    break;
                }
                Key::Char('a') => {
                    moving_selection = move_selection_left(
                        &moving_selection,
                        &window,
                        (ts.0, ts.1 - 1), // TODO use variable for this
                        1);
                }
                Key::Char('d') => {
                    moving_selection = move_selection_right(
                        &moving_selection,
                        &window,
                        (ts.0, ts.1 - 1), // TODO use variable for this
                        1);
                }
                Key::Char('w') => {
                    moving_selection = move_selection_up(
                        &moving_selection,
                        &window,
                        (ts.0, ts.1 - 1), // TODO use variable for this
                        1);
                }
                Key::Char('s') => {
                    moving_selection = move_selection_down(
                        &moving_selection,
                        &window,
                        (ts.0, ts.1 - 1), // TODO use variable for this
                        1);
                }
                Key::Char('q') => {
                }
                _ => {}
            },
            // Event::Mouse(me) => match me {
            //     MouseEvent::Press(_, x, y) => {
            //         write!(stdout, "{}x", termion::cursor::Goto(x, y)).unwrap();
            //     }
            //     _ => (),
            // },
            _ => {}
        }
        terminal_render(
            &pixels,
            bounds,
            &window,
            &moving_selection,
            (ts.0, ts.1 - 1),
            (0, 1),
        );
        stdout.flush().unwrap();
    }
}

/// Given a pixel array, bounds and window co-ordinates and number of threads to use
/// render to the pixel array
fn parallel_render(
    pixels: &mut Vec<u8>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
    num_threads: usize,
) {
    println!("Parallel render with {:?} threads", num_threads);

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

/// Given a rendered Mandelbrot image of the size bounds calculate the value of the terminal
/// representation. Given the terminal width and height and a character position, you must find
/// the average brightness of the pixels that character represents. Note that there are 24
/// gray scales. char_pos is 1 based
fn pixel_to_char_grayscale(
    char_pos: (u16, u16),
    terminal_bounds: (u16, u16),
    pixels: &Vec<u8>,
    bounds: (usize, usize),
) -> u8 {
    let horizontal_pixels_per_char = bounds.0 as f32 / terminal_bounds.0 as f32;
    let vertical_pixels_per_char = bounds.1 as f32 / terminal_bounds.1 as f32;

    let leftmost_pixel = (char_pos.0 - 1) as f32 * horizontal_pixels_per_char;
    let topmost_pixel = (char_pos.1 - 1) as f32 * vertical_pixels_per_char;

    let rightmost_pixel = (leftmost_pixel + horizontal_pixels_per_char) as usize;
    let bottommost_pixel = (topmost_pixel + vertical_pixels_per_char) as usize;

    let mut sum: usize = 0;
    let mut count: usize = 0;
    for x in leftmost_pixel as usize..rightmost_pixel {
        for y in topmost_pixel as usize..bottommost_pixel {
            sum += pixels[x + y * bounds.0] as usize;
            count += 1;
        }
    }

    let avg = sum as f32 / count as f32;
    let gray_scale: f32 = 24.0 / 256.0;
    let ret = (avg * gray_scale) as u8;
    return ret;
}

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

fn write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (usize, usize),
) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;
    let encoder = PNGEncoder::new(output);
    encoder.encode(pixels, bounds.0 as u32, bounds.1 as u32, ColorType::Gray(8))?;
    Ok(())
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
    let (r, c) = complex_to_cursor_position(&selection.upper_left, &window, terminal_size);
    assert_eq!((r, c), (50, 50));
}
