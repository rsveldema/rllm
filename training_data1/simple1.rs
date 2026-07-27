use std::collections::HashMap;
use std::fmt::{self, Display};

const DEFAULT_LIMIT: usize = 4;
static APPLICATION_NAME: &str = "rust-keywords";

type Identifier = u64;
type NameMap = HashMap<Identifier, String>;

#[derive(Clone, Debug)]
struct User {
    id: Identifier,
    name: String,
    active: bool,
}

#[derive(Debug)]
enum LookupError {
    Missing(Identifier),
    Inactive(String),
}

union IntegerBits {
    signed: i32,
    unsigned: u32,
}

trait Describe {
    fn describe(&self) -> String;
}

impl Describe for User {
    fn describe(&self) -> String {
        format!("{}#{}", self.name, self.id)
    }
}

impl Display for LookupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(identifier) => write!(formatter, "missing {identifier}"),
            Self::Inactive(name) => write!(formatter, "inactive {name}"),
        }
    }
}

fn find_user(users: &NameMap, identifier: Identifier) -> Result<&str, LookupError> {
    match users.get(&identifier) {
        Some(name) if !name.is_empty() => Ok(name.as_str()),
        Some(name) => Err(LookupError::Inactive(name.clone())),
        None => Err(LookupError::Missing(identifier)),
    }
}

fn collect_positive(values: &[i32]) -> Vec<i32> {
    let mut result = Vec::new();
    for &value in values {
        if value <= 0 {
            continue;
        }
        result.push(value);
    }
    result
}

fn sum_until_limit(values: &[i32], limit: i32) -> i32 {
    let mut index = 0;
    let mut sum = 0;
    while index < values.len() {
        sum += values[index];
        if sum >= limit {
            break;
        }
        index += 1;
    }
    sum
}

fn first_even(values: &[i32]) -> Option<i32> {
    for &value in values {
        if value % 2 == 0 {
            return Some(value);
        }
    }
    None
}

fn countdown(mut value: i32) -> Vec<i32> {
    let mut result = Vec::new();
    loop {
        if value <= 0 {
            break;
        }
        result.push(value);
        value -= 1;
    }
    result
}

fn apply_description<T>(value: &T) -> String
where
    T: Describe + ?Sized,
{
    value.describe()
}

fn dynamic_description(value: &dyn Describe) -> String {
    value.describe()
}

fn make_multiplier(factor: i32) -> impl Fn(i32) -> i32 {
    move |value| value * factor
}

unsafe fn read_union(bits: IntegerBits) -> u32 {
    unsafe { bits.unsigned }
}

extern "C" fn add(left: i32, right: i32) -> i32 {
    left + right
}

async fn asynchronous_value(value: i32) -> i32 {
    value
}

async fn await_value(value: i32) -> i32 {
    asynchronous_value(value).await
}

mod utilities {
    pub fn is_positive(value: i32) -> bool {
        value > 0
    }

    pub mod nested {
        pub fn application_name() -> &'static str {
            super::super::APPLICATION_NAME
        }
    }
}

fn main() {
    let mut names = NameMap::new();
    names.insert(1, String::from("Ada"));
    names.insert(2, String::from("Grace"));

    let user = User {
        id: 1,
        name: String::from("Ferris"),
        active: true,
    };
    let values = [-3, 0, 2, 7];
    let multiply = make_multiplier(3);
    let bits = IntegerBits { signed: -1 };

    println!("{}", find_user(&names, 1).unwrap_or("missing"));
    println!("{:?}", collect_positive(&values));
    println!("{}", sum_until_limit(&values, DEFAULT_LIMIT as i32));
    println!("{:?}", first_even(&values));
    println!("{:?}", countdown(3));
    println!("{}", apply_description(&user));
    println!("{}", dynamic_description(&user));
    println!("{}", multiply(4));
    println!("{}", add(2, 5));
    println!("{}", utilities::is_positive(8));
    println!("{}", utilities::nested::application_name());
    println!("{}", user.active);

    unsafe {
        println!("{}", read_union(bits));
    }

    let _future = await_value(9);
}
