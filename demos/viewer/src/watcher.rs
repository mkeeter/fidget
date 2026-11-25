use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use log::{debug, warn};
use std::path::Path;

/// Watches for changes to the given file and sends it on `tx`
pub(crate) fn file_watcher_thread(
    path: &Path,
    rx: Receiver<()>,
    tx: Sender<(String, String)>,
) -> Result<()> {
    let read_file = || -> Result<String> {
        let out = String::from_utf8(std::fs::read(path)?).unwrap();
        Ok(out)
    };
    let mut contents = read_file()?;
    let path_str = path.to_string_lossy().to_string();
    tx.send((contents.clone(), path_str.clone()))?;

    loop {
        // Wait for a file change notification
        rx.recv()?;
        let new_contents = loop {
            match read_file() {
                Ok(c) => break c,
                Err(e) => {
                    warn!("file read error: {e:?}");
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        };
        if contents != new_contents {
            contents = new_contents;
            debug!("file contents changed!");
            tx.send((contents.clone(), path_str.clone()))?;
        }
    }
}
