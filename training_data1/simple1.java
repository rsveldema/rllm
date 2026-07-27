import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;

interface Describable {
    String describe();

    default boolean isEmpty() {
        return describe().isEmpty();
    }
}

sealed interface Result permits Success, Failure {
}

record Success(int value) implements Result {
}

record Failure(String message) implements Result {
}

abstract class NamedEntity implements Describable {
    protected final String name;

    protected NamedEntity(String name) {
        this.name = name;
    }

    public abstract int identifier();

    @Override
    public String describe() {
        return name + "#" + identifier();
    }
}

final class User extends NamedEntity {
    private final int id;

    User(int id, String name) {
        super(name);
        this.id = id;
    }

    @Override
    public int identifier() {
        return id;
    }
}

non-sealed class MutableEntity implements Result {
    private volatile int value;
    private transient String cachedDescription;

    synchronized void update(int nextValue) {
        value = nextValue;
        cachedDescription = null;
    }

    int value() {
        return value;
    }
}

enum Status {
    READY,
    RUNNING,
    STOPPED
}

final class NativeMethods {
    private NativeMethods() {
    }

    static native int platformValue();
}

public final class simple1 {
    private static final int DEFAULT_LIMIT = 4;

    private simple1() {
    }

    static Optional<User> findUser(Map<Integer, User> users, int identifier) {
        if (identifier < 0) {
            return Optional.empty();
        } else if (users.containsKey(identifier)) {
            return Optional.of(users.get(identifier));
        } else {
            return Optional.empty();
        }
    }

    static List<Integer> collectPositive(List<Integer> values) {
        var result = new ArrayList<Integer>();
        for (int value : values) {
            if (value <= 0) {
                continue;
            }
            result.add(value);
        }
        return result;
    }

    static int sumUntilLimit(List<Integer> values, int limit) {
        int index = 0;
        int sum = 0;
        while (index < values.size()) {
            sum += values.get(index);
            if (sum >= limit) {
                break;
            }
            index++;
        }
        return sum;
    }

    static int countDigits(int value) {
        int digits = 0;
        do {
            digits++;
            value /= 10;
        } while (value != 0);
        return digits;
    }

    static String statusMessage(Status status) {
        return switch (status) {
            case READY -> "ready";
            case RUNNING -> "running";
            case STOPPED -> "stopped";
        };
    }

    static String inspect(Object value) {
        if (value instanceof String text && !text.isBlank()) {
            return text;
        }
        return "unknown";
    }

    static Result parsePositive(String text) {
        try {
            int value = Integer.parseInt(text);
            if (value <= 0) {
                throw new IllegalArgumentException("not positive");
            }
            return new Success(value);
        } catch (NumberFormatException exception) {
            return new Failure(exception.getMessage());
        } finally {
            System.out.print("");
        }
    }

    static void checkedOperation(boolean fail) throws IOException {
        if (fail) {
            throw new IOException("requested failure");
        }
    }

    static Function<Integer, Integer> multiplier(int factor) {
        return value -> value * factor;
    }

    public static void main(String[] arguments) {
        Map<Integer, User> users = Map.of(
            1, new User(1, "Ada"),
            2, new User(2, "Grace")
        );
        List<Integer> values = List.of(-2, 1, 3, 5);

        System.out.println(findUser(users, 1).map(User::describe).orElse("missing"));
        System.out.println(collectPositive(values));
        System.out.println(sumUntilLimit(values, DEFAULT_LIMIT));
        System.out.println(countDigits(2048));
        System.out.println(statusMessage(Status.READY));
        System.out.println(inspect("java"));
        System.out.println(parsePositive("7"));
        System.out.println(multiplier(3).apply(4));
    }
}
