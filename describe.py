import pandas as pd
import argparse



def count(col):
    count = []
    for col in cols:
        count.append(int(df[col].count()))
    return count

def mean(col):
    means = []
    for col in cols:
        sum_val = 0.0
        count_val = 0

        for val in df[col]:
            if pd.isna(val):
                continue

            try:
                num = float(val)
                sum_val += num
                count_val += 1
            except (ValueError, TypeError):
                continue

        if count_val == 0:
            means.append('NaN')
        else:
            means.append(sum_val / count_val)
    return means

def colmin(cols, df):
    min_vals = []
    for col in cols:
        numeric_values = []
        for val in df[col]:
            if val is None or pd.isna(val):
                continue
            try:
                numeric_values.append(float(val))
            except ValueError:
                continue
        min_vals.append(min(numeric_values) if numeric_values else None)
    return min_vals

def colmax(cols, df):
    max_vals = []
    for col in cols:
        numeric_values = []
        for val in df[col]:
            if val is None or pd.isna(val):
                continue
            try:
                numeric_values.append(float(val))
            except ValueError:
                continue
        max_vals.append(max(numeric_values) if numeric_values else None)
    return max_vals


def quantiles(df, cols):
    q25_vals = []
    q50_vals = []
    q75_vals = []

    for col in cols:
        numeric_values = []
        for val in df[col]:
            if pd.isna(val):
                continue
            try:
                numeric_values.append(float(val))
            except (ValueError, TypeError):
                continue

        if numeric_values:
            sorted_vals = sorted(numeric_values)
            n = len(sorted_vals)

            def percentile(p):
                k = (n - 1) * p
                f = int(k)
                c = min(f + 1, n - 1)
                return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

            q25_vals.append(percentile(0.25))
            q50_vals.append(percentile(0.50))
            q75_vals.append(percentile(0.75))
        else:
            q25_vals.append(None)
            q50_vals.append(None)
            q75_vals.append(None)
    return q25_vals, q50_vals, q75_vals

def std(df, cols, means):
    std_vals = []
    for i, col in enumerate(cols):
        numeric_values = 0.0
        n = 0.0
        for val in df[col]:
            if pd.isna(val):
                continue
            try:
                numeric_values+= (val - means[i]) ** 2
                n += 1
            except (ValueError, TypeError):
                continue

        if numeric_values:
            variance = numeric_values / n
            std_vals.append(variance ** 0.5)
        else:
            std_vals.append(None)
    return std_vals


def top_and_freq(df, cols):
    tops = []
    freqs = []

    for col in cols:
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            tops.append(None)
            freqs.append(0)
            continue

        mode_vals = series.mode()
        if mode_vals.empty:
            tops.append(None)
            freqs.append(0)
            continue

        top_val = mode_vals.iloc[0]
        freq_val = (series == top_val).sum()

        tops.append(top_val)
        freqs.append(freq_val)

    return tops, freqs


def print_summary(df, cols, rows):
    headers = cols

    colw = max(12, max((len(h) for h in headers), default=12) + 2)

    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == 'NaN':
            return "NaN"
        try:
            if isinstance(v, (int, float)):
                return f"{float(v):.6f}"
            return f"{float(v):.6f}"
        except Exception:
            return str(v)

    print(f"{'':<12}", end="")
    for h in headers:
        print(f"{h:>{colw}}", end="")
    print()

    for label, seq in rows:
        print(f"{label:<12}", end="")
        for i, col in enumerate(headers):
            val = seq[i] if i < len(seq) else None
            print(f"{fmt(val):>{colw}}", end="")
        print()


    
if '__main__' == __name__:
    # get arguments
    parser = argparse.ArgumentParser(description="Train logistic regression")
    parser.add_argument("input_file", nargs="?", default="datasets/dataset_train.csv",
                        help="path to training CSV file (default: datasets/dataset_train.csv)")
    args = parser.parse_args()
    df = pd.read_csv(args.input_file)
    # compute statistics
    cols = df.columns.tolist()

    counts = count(cols)
    means = mean(cols)
    mins = colmin(cols, df)
    maxs = colmax(cols, df)
    stds = std(df, cols, means)
    q25, q50, q75 = quantiles(df, cols)
    top_val, freq_val = top_and_freq(df, cols)
    
    # print summary
    rows = [
        ("Count", counts),
        ("Mean",  means),
        ("Std",   stds),
        ("Min",   mins),
        ("25%",   q25),
        ("50%",   q50),
        ("75%",   q75),
        ("Max",   maxs),
        ("Top",   top_val),
        ("Freq",  freq_val),
    ]
    
    print_summary(df, cols, rows)
    