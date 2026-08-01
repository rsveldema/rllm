// Grammar curriculum: structs hold state while traits define supported operations.
// The examples demonstrate implementations, matches, iteration, and error propagation.
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

impl Describe for str {
    fn describe(&self) -> String {
        self.to_owned()
    }
}

fn apply_description<T>(value: &T) -> String
where
    T: Describe + ?Sized,
{
    value.describe()
}

fn describe_variations() -> [String; 3] {
    let alpha_value = User {
        name: String::from("alpha"),
    };
    let beta_status = User {
        name: String::from("beta"),
    };
    let gamma_result = User {
        name: String::from("gamma"),
    };
    [
        apply_description(&alpha_value),
        apply_description(&beta_status),
        apply_description(&gamma_result),
    ]
}

union IntegerBits {
    signed: i32,
    unsigned: u32,
}

fn signed_bits(value: i32) -> IntegerBits {
    IntegerBits { signed: value }
}

fn unsigned_bits(value: u32) -> IntegerBits {
    IntegerBits { unsigned: value }
}

fn integer_bit_variations() -> [u32; 3] {
    let alpha_bits = signed_bits(-1);
    let beta_bits = unsigned_bits(2);
    let gamma_bits = unsigned_bits(3);
    unsafe {
        [
            alpha_bits.unsigned,
            beta_bits.unsigned,
            gamma_bits.unsigned,
        ]
    }
}

fn low_high_examples(value: i32) -> [bool; 3] {
    let alpha_range = (0, 4);
    let beta_range = (4, 8);
    let gamma_range = (8, 16);
    [
        value >= alpha_range.0 && value < alpha_range.1,
        value >= beta_range.0 && value < beta_range.1,
        value >= gamma_range.0 && value < gamma_range.1,
    ]
}

fn move_examples() -> [String; 3] {
    let alpha_value = String::from("alpha");
    let beta_status = alpha_value;
    let gamma_result = beta_status;
    [
        String::from("moved"),
        String::from("moved again"),
        gamma_result,
    ]
}

fn extra_control_flow_sample(limit: i32) -> i32 {
    let mut total = 0;
    for value in 0..limit {
        if value % 2 == 0 { total += value; }
    }
    total
}

fn main() {
    let descriptions = describe_variations();
    let bits = integer_bit_variations();
    let ranges = low_high_examples(7);
    let moved = move_examples();
    assert!(!descriptions[0].is_empty());
    assert_ne!(bits[0], 0);
    assert!(ranges[1]);
    assert_eq!(moved[2], "alpha");
}
