// embedding
use std::io::{self, Write};

pub fn read_stdin(render: &str) -> io::Result<String>{
    print!("{render}");
    io::stdout().flush()?;

    let mut buffer=String::new();
    io::stdin().read_line(&mut buffer)?;

    if buffer.ends_with('\n'){
        buffer.pop();
        if buffer.ends_with('\r'){
            buffer.pop();
        }
    }

    Ok(buffer)
}
