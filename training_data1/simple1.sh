#!/usr/bin/env bash

set -euo pipefail

readonly script_name="${0##*/}"
declare -a values=(1 2 3 4)
declare -A labels=([primary]="main" [secondary]="backup")

function print_message() {
    local message="${1:-empty}"
    printf '%s: %s\n' "${script_name}" "${message}"
}

is_positive() {
    local value="$1"
    [[ "${value}" =~ ^[0-9]+$ ]] && (( value > 0 ))
}

describe_value() {
    local value="$1"
    if (( value < 0 )); then
        printf '%s\n' "negative"
    elif (( value == 0 )); then
        printf '%s\n' "zero"
    else
        printf '%s\n' "positive"
    fi
}

case "${1:-start}" in
    start | run)
        print_message "starting"
        ;;
    stop)
        print_message "stopping"
        ;;
    status)
        print_message "ready"
        ;&
    *)
        print_message "unknown command"
        ;;
esac

for value in "${values[@]}"; do
    if is_positive "${value}"; then
        printf 'positive value: %d\n' "${value}"
    else
        continue
    fi
done

for ((index = 0; index < ${#values[@]}; ++index)); do
    printf 'value[%d]=%d\n' "${index}" "${values[index]}"
done

counter=3
while (( counter > 0 )); do
    printf 'countdown: %d\n' "${counter}"
    ((counter -= 1))
done

attempt=0
until (( attempt >= 2 )); do
    ((attempt += 1))
    printf 'attempt: %d\n' "${attempt}"
done

select choice in primary secondary quit; do
    case "${choice}" in
        primary | secondary)
            printf 'selected: %s\n' "${labels[${choice}]}"
            break
            ;;
        quit)
            break
            ;;
        *)
            printf '%s\n' "invalid selection"
            ;;
    esac
done < /dev/null

if [[ -n "${labels[primary]}" && -v 'labels[secondary]' ]]; then
    print_message "labels are configured"
fi

if [[ -f "${BASH_SOURCE[0]}" || -L "${BASH_SOURCE[0]}" ]]; then
    print_message "source path exists"
fi

! false
true && print_message "and-list succeeded"
false || print_message "or-list recovered"

{
    printf '%s\n' "group command"
    printf '%s\n' "current shell"
} > /dev/null

(
    temporary_value="subshell"
    printf '%s\n' "${temporary_value}"
) | while IFS= read -r line; do
    printf 'pipeline: %s\n' "${line}"
done

command_output="$(
    printf '%s\n' "command substitution"
)"
arithmetic_result=$((values[0] + values[1]))
fallback_value="${UNSET_VALUE:-fallback}"
assigned_value="${ANOTHER_VALUE:=assigned}"
prefix_removed="${script_name#simple}"
suffix_removed="${script_name%.sh}"

printf '%s\n' \
    "${command_output}" \
    "${arithmetic_result}" \
    "${fallback_value}" \
    "${assigned_value}" \
    "${prefix_removed}" \
    "${suffix_removed}"

read -r first second <<< "alpha beta"
printf 'first=%s second=%s\n' "${first}" "${second}"

while IFS= read -r item; do
    printf 'document item: %s\n' "${item}"
done <<'ITEMS'
one
two
three
ITEMS

coproc PRODUCER {
    printf '%s\n' "background result"
}

if read -r coprocess_result <&"${PRODUCER[0]}"; then
    printf 'coprocess: %s\n' "${coprocess_result}"
fi
wait "${PRODUCER_PID}"

time {
    describe_value 0
    describe_value 7
} 2> /dev/null

printf -v formatted_value 'formatted-%03d' 7
export formatted_value
unset command_output

return_from_function() {
    local status="${1:-0}"
    return "${status}"
}

if return_from_function 0; then
    print_message "function returned successfully"
fi

exit 0
