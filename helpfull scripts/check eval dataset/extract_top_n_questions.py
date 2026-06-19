import csv
import pandas as pd

# Method 1: Using pandas (recommended)
def extract_top_rows_pandas(input_file, output_file, num_rows=55):
    """
    Extract first num_rows from CSV file using pandas
    Keeps original data exactly as is
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Get first 55 rows
        top_rows = df.head(num_rows)
        
        # Save to new CSV file without adding any new columns
        top_rows.to_csv(output_file, index=False)
        
        print(f"✅ Successfully created '{output_file}' with {len(top_rows)} rows")
        print(f"📊 Original file had {len(df)} rows")
        print(f"📋 Original data preserved - no new columns added")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
    except Exception as e:
        print(f"❌ Error: {e}")

# Method 2: Using csv module (no external dependencies)
def extract_top_rows_csv(input_file, output_file, num_rows=55):
    """
    Extract first num_rows from CSV file using built-in csv module
    Keeps original data exactly as is
    """
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # Read header
            header = next(reader)
            
            # Read first 55 rows
            rows = []
            counter = 0
            for row in reader:
                if counter < num_rows:
                    rows.append(row)
                    counter += 1
                else:
                    break
        
        # Write to new CSV file - exactly as original
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Write header (no new columns added)
            writer.writerow(header)
            
            # Write rows (no counter added)
            for row in rows:
                writer.writerow(row)
        
        print(f"✅ Successfully created '{output_file}' with {len(rows)} rows")
        print(f"📋 Original data preserved - no new columns added")
        
    except FileNotFoundError:
        print(f"❌ Error: File '{input_file}' not found")
    except StopIteration:
        print(f"⚠️  Warning: CSV file has no header row")
    except Exception as e:
        print(f"❌ Error: {e}")

# Usage example
if __name__ == "__main__":
    # Specify your input file name
    input_filename = "mistakes.csv"  # Change this to your actual filename
    output_filename = "mistakes_top_55.csv"
    
    # Choose one of the methods below:
    
    # Method 1: Using pandas (recommended for large files)
    extract_top_rows_pandas(input_filename, output_filename)
    
    # Method 2: Using built-in csv module (no pandas required)
    # extract_top_rows_csv(input_filename, output_filename)