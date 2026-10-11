use std::error::Error;
mod mnist;

fn main() -> Result<(), Box<dyn Error>> {
    {
        println!("Hello, world!");

        let train = mnist::load_csv("data/mnist_train.csv")?;
        println!("{} images", train.len());

        for row in train.image(0).chunks(28) {
            let line: String = row.iter().map(|&p| if p > 0.5 { '#' } else { '.' }).collect();
            println!("{line}");
        }
        println!("label: {}", train.labels[0]);

        Ok(())
    }
}
