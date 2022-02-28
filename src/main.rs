use image::png::PNGEncoder;
use image::ColorType;
use num::Complex;
use std::env;
use std::fs::File;
use std::io::{stdin, stdout, Write};
use std::str::FromStr;
use termion::event::{Event, Key, MouseEvent};
use termion::input::{MouseTerminal, TermRead};
use termion::raw::IntoRawMode;
use termion::{color, style};

fn main() {
    let args = env::args().collect::<Vec<String>>();

    let num_cpus = num_cpus::get();
    println!("Found {:?} cpus.", num_cpus);

    if args.len() < 5 {
        eprintln!(
            "Usage: {} FILE PIXELS UPPERLEFT UPPERRIGHT [THREADS]",
            args[0]
        );
        eprintln!(
            "Example: {} mandel.png 1280x768 -1.20,0.35 -1,0.20 8",
            args[0]
        );
        std::process::exit(1);
    }

    let bounds = parse_pair(&args[2], 'x').expect("error parsing image dimensions");
    let upper_left = parse_complex(&args[3]).expect("error parsing upper left corner point");
    let lower_right = parse_complex(&args[4]).expect("error parsing lower right corner point");
    let mut pixels = vec![0; bounds.0 * bounds.1];

    // Handle thread input
    let num_threads: usize = if args.len() == 6 {
        args[5].parse().unwrap()
    } else {
        num_cpus
    };

    assert!(num_threads > 0);

    // println!("{}Red", color::Fg(color::Red));
    // println!("{}Blue", color::Fg(color::Blue));
    // println!("{}Blue'n'Bold{}", style::Bold, style::Reset);
    // println!("{}Just plain italic", style::Italic);
    // print!("{}Stuff", termion::cursor::Goto(1, 1));

    parallel_render(&mut pixels, bounds, upper_left, lower_right, num_threads);

    tui_loop(&mut pixels, bounds, upper_left, lower_right);

    //     write_image(&args[1], &pixels, bounds).expect("error writing PNG file");
}

/// Draw the pixels of the Mandelbrot set as best we can in the terminal
/// https://www.unicode.org/charts/PDF/U2580.pdf
fn render_to_terminal(
    pixels: &Vec<u8>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    let (cols, rows) = termion::terminal_size().unwrap();
    const gray_shades: f32 = 24.0;

    for c in 1..cols {
        for r in 1..rows {
            let pixel = pixel_to_char_grayscale((c,r), (cols,rows), pixels, bounds);

            println!(
                "{}{}\u{2588}",
                termion::cursor::Goto(c, r),
                termion::color::Fg(termion::color::AnsiValue::grayscale(pixel))
            );
        }
    }
}

/// Render to the terminal
fn tui_loop(
    pixels: &mut Vec<u8>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    let stdin = stdin();
    let mut stdout = MouseTerminal::from(stdout().into_raw_mode().unwrap());

    let ts = termion::terminal_size().unwrap();

    write!(
        stdout,
        "{}{}q to exit. Click, click, click!",
        termion::clear::All,
        termion::cursor::Goto(1, 1)
    )
    .unwrap();

    render_to_terminal(&pixels, bounds, upper_left, lower_right);

    stdout.flush().unwrap();

    for c in stdin.events() {
        let evt = c.unwrap();
        match evt {
            Event::Key(Key::Char('q')) => break,
            Event::Mouse(me) => match me {
                MouseEvent::Press(_, x, y) => {
                    write!(stdout, "{}x", termion::cursor::Goto(x, y)).unwrap();
                }
                _ => (),
            },
            _ => {}
        }
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
/// gray scales.
fn pixel_to_char_grayscale(
    char_pos: (u16, u16),
    terminal_bounds: (u16, u16),
    pixels: &Vec<u8>,
    bounds: (usize, usize),
) -> u8 {
    let horizontal_pixels_per_char = terminal_bounds.0 as f32 / bounds.0 as f32;
    let vertical_pixels_per_char = terminal_bounds.1 as f32 / bounds.1 as f32;

    let leftmost_pixel = char_pos.0 as f32 * horizontal_pixels_per_char;
    let topmost_pixel = char_pos.1 as f32 * vertical_pixels_per_char;

    let rightmost_pixel = (leftmost_pixel + horizontal_pixels_per_char) as usize;
    let bottommost_pixel = (topmost_pixel + vertical_pixels_per_char) as usize;

    let mut sum: usize = 0;
    let mut count: usize = 0;
    for x in leftmost_pixel as usize..rightmost_pixel {
        for y in topmost_pixel as usize..bottommost_pixel {
            sum += pixels[x + y * bounds.1] as usize;
            count += 1;
        }
    }

    let ret = ((sum as f32 / count as f32) * (24.0 / 256.0)) as u8;
    // println!("{:?} {:?} {:?}", sum, count, ret);
    return ret
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
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64, // Why subtraction here? pixel.1 increases as we go down,
                                                                       // but the imaginary component increases as we go up.
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
