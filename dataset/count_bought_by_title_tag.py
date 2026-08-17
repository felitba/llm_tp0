import csv
import re
from collections import defaultdict


def count_bought_by_title_tag(csv_filepath):
    # Dictionary to hold the counts: { "(Tag)": {"true": 0, "false": 0} }
    tag_counts = defaultdict(lambda: {"true": 0, "false": 0})

    # Regex pattern to match "(Some Text)" only at the very end of the string
    # \(     -> matches literal opening parenthesis
    # [^)]+  -> matches one or more characters that are NOT a closing parenthesis
    # \)     -> matches literal closing parenthesis
    # $      -> ensures this is at the end of the string
    pattern = re.compile(r'(\([^)]+\))$')

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                title = row.get('title', '').strip()

                # Normalize the 'bought' column to lowercase 'true' or 'false'
                bought_status = row.get('bought', '').strip().lower()

                # Search for the pattern at the end of the title
                match = pattern.search(title)
                if match:
                    tag = match.group(1)  # Extracts the "(Some Text)"
                else:
                    tag = "(No Tag Found)"  # Fallback for titles without the suffix

                # Increment the correct counter based on the bought column
                if bought_status == 'true':
                    tag_counts[tag]["true"] += 1
                elif bought_status == 'false':
                    tag_counts[tag]["false"] += 1

        # Display the results in a clean table format
        print(f"{'Title Tag':<25} | {'Bought (True)':<15} | {'Bought (False)':<15}")
        print("-" * 62)
        sorted_tag_counts = sorted(tag_counts.items(), key=lambda x: x[1]['true'], reverse=True)  # Sort by tag for consistent output
        for tag, counts in sorted_tag_counts:
            print(f"{tag:<25} | {counts['true']:<15} | {counts['false']:<15}")

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    # Example usage
    count_bought_by_title_tag('supermarket_products.csv')