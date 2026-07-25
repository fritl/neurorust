use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::Cursor;
use std::io::Read;

use flate2::read::GzDecoder;

#[derive(Debug)]
pub struct MnistData {
    pub sizes: Vec<i32>,
    pub data: Vec<u8>,
}

impl MnistData {
    pub fn new(f: &File) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents = Vec::new();
        gz.read_to_end(&mut contents)?;

        let mut r = Cursor::new(&contents);
        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        let magic_number = r.read_i32::<BigEndian>()?;

        match magic_number {
            2049 => sizes.push(r.read_i32::<BigEndian>()?),
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!("Magic number must be 2049 or 2051"),
        };
        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }

    pub fn from_bytes(f: &[u8]) -> Result<MnistData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents = Vec::new();
        gz.read_to_end(&mut contents)?;

        let mut r = Cursor::new(&contents);
        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();
        let magic_number = r.read_i32::<BigEndian>()?;

        match magic_number {
            2049 => sizes.push(r.read_i32::<BigEndian>()?),
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!("Magic number must be 2049 or 2051"),
        };
        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}
