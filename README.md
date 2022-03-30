# rust-mandelbrot
This is an extension of the code you can find in the book Programming Rust: Fast, Safe Systems Development that renders Mandelbrot sets to png files. I've add a TUI interface to enable viewing the image in the terminal, positioning and zooming into the image and the ability to save the current image with a key press.

## Running

Currently, just like the book example, you pass in a filename, image size and window into the set using two complex numbers for the top left and bottom right corners respectively.

```
cargo r --release -- /tmp/mandle.png 1280x768 -1.20,0.35 -1.0,0.20
```

## TODO list
Next...

pass initial filename into tui loop 
when user hits save get the next filename and save it

when the user hits zoom reset the window coordinates 
redraw the pixels array 
draw the tui screen


All...

Function to map from Complex coord to screen cursor

Let the user move visible selection window around with the mouse
  Clamp it to the screen area
  It should have size and aspect ratio of pixels 
  It should start at half the current selected area

When user presses a particular key zoom in and redraw 

Add two keys for users to change the zoom level

Add a key to save the current image appending a number to the given filename




