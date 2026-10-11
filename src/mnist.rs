use std::error::Error;
use std::fs;

const PIXELS: usize = 784;

pub struct Dataset {
    pub images: Vec<f32>,
    pub labels: Vec<u8>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn image(&self, i: usize) -> &[f32] {
        &self.images[i * 784..(i + 1) * 784]
    }
}

pub fn load_csv(path: &str) -> Result<Dataset, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    if content.is_empty() {
        return Err(format!("{path}: file is empty").into());
    }

    let mut images: Vec<f32> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    for (i, line) in content.lines().enumerate().skip(1) {
        let line_no = i + 1;

        if line.trim().is_empty() {
            continue;
        }

        let mut fields = line.split(',');

        let label: u8 = fields
            .next()
            .ok_or_else(|| format!("line {line_no}: missing label"))?
            .trim()
            .parse()
            .map_err(|e| format!("line {line_no}: bad label: {e}"))?;
        if label > 9 {
            return Err(format!("line {line_no}: label {label} out of range").into());
        }

        let start = images.len();
        for field in fields {
            let px: u8 = field
                .trim()
                .parse()
                .map_err(|e| format!("line {line_no}: bad pixel '{field}': {e}"))?;
            images.push(px as f32 / 255.0);
        }

        let count = images.len() - start;
        if count != PIXELS {
            return Err(format!("line {line_no}: expected {PIXELS} pixels, got {count}").into());
        }

        labels.push(label);
    }

    if labels.is_empty() {
        return Err(format!("{path}: no data rows").into());
    }

    Ok(Dataset { images, labels })
}