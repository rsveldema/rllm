use std::fmt::{self, Display};

#[derive(Debug)]
enum LookupError {
    Missing,
    Inactive(String),
}

union IntegerBits {
    signed: i32,
    unsigned: u32,
}

trait Describe {
    fn describe(&self) -> String;
}

struct User {
    name: String,
}

impl Describe for User {
    fn describe(&self) -> String {
        self.name.clone()
    }
}

impl Display for User {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.name)
    }
}

fn inactive_name(name: String) -> Result<String, LookupError> {
    Some(name).ok_or(LookupError::Missing).and_then(|name| Err(LookupError::Inactive(name)))
}

fn inactive_copy(name: &str) -> Result<String, LookupError> {
    Some(name).ok_or(LookupError::Missing).and_then(|name| Err(LookupError::Inactive(name.to_string())))
}

fn signed_bits(value: i32) -> IntegerBits {
    IntegerBits { signed: value }
}

fn unsigned_bits(value: u32) -> IntegerBits {
    IntegerBits { unsigned: value }
}

fn describe_value<T: Describe>(value: &T) -> String {
    value.describe()
}

fn describe_pair<T: Describe>(first: &T, second: &T) -> String {
    format!("{} {}", first.describe(), second.describe())
}

fn main() {
    let user = User { name: String::from("Ada") };
    let bits = IntegerBits { signed: 7 };
    let other_bits = IntegerBits { unsigned: 9 };
    let _ = inactive_name(String::from("offline"));
    let _ = inactive_copy("away");
    let _ = signed_bits(3);
    let _ = unsigned_bits(5);
    println!("{}", describe_value(&user));
    println!("{}", describe_pair(&user, &user));
    let _ = bits;
    let _ = other_bits;
}
