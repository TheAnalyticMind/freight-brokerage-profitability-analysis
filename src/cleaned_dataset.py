import pandas as pd
import numpy as np

from pathlib import Path


path = Path(__file__).resolve().parent


# Read the raw Excel file into a DataFrame
df = pd.read_excel(path / 'input' / 'Sales_Assessment_Data.xlsx')
print(f"Raw dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ============================================================
# ISSUE 1: DUPLICATE SHIPMENT IDs
# Problem: 21 duplicate Shipment IDs found — same ID appears
#          across multiple rows, sometimes with different dates
#          and status (e.g., Quote + Delivered), suggesting
#          a shipment progressed through lifecycle stages.
# Fix: Retain the most recent / highest-status row per Shipment ID.
#      Status priority: Delivered > Complete > In Transit > Dispatched
#      > Committed > Quote > Canceled
# ============================================================

# Define status priority for deduplication (lower number = keep first)
STATUS_PRIORITY = {
    'Delivered': 1,
    'Complete': 2,
    'In Transit': 3,
    'Out for Delivery': 4,
    'Dispatched': 5,
    'Committed': 6,
    'Quote': 7,
    'Canceled': 8
}

def normalize_status_for_priority(s):
    """Map any raw status string to a canonical group for priority sorting."""
    if pd.isna(s):
        return 'Quote'
    s_lower = str(s).strip().lower()
    if s_lower in ['delivered', 'deliverd', 'deliver', 'dlvrd']:
        return 'Delivered'
    elif s_lower in ['complete', 'completed']:
        return 'Complete'
    elif s_lower in ['in transit', 'in-transit', 'intransit']:
        return 'In Transit'
    elif s_lower in ['out for delivery', 'ofd']:
        return 'Out for Delivery'
    elif s_lower == 'dispatched':
        return 'Dispatched'
    elif s_lower == 'committed':
        return 'Committed'
    elif s_lower in ['quote', 'quoted', 'ready', 'sent']:
        return 'Quote'
    elif s_lower in ['canceled', 'cancelled', 'cncld', 'cancled']:
        return 'Canceled'
    return 'Quote'

# Assign priority score for deduplication sorting
df['_status_priority'] = df['Status'].apply(
    lambda s: STATUS_PRIORITY.get(normalize_status_for_priority(s), 9)
)

# Sort so highest-priority status (lowest number) comes first per Shipment ID
df.sort_values(['Shipment ID', '_status_priority', 'Pickup Date'],
               ascending=[True, True, False], inplace=True)

# Keep only the first (highest-priority, most recent) row per Shipment ID
before_dedup = len(df)
df = df.drop_duplicates(subset='Shipment ID', keep='first')
after_dedup = len(df)
print(f"Issue 1 - Duplicates removed: {before_dedup - after_dedup} rows dropped")

# Drop the helper column
df.drop(columns=['_status_priority'], inplace=True)

# ============================================================
# ISSUE 2: STATUS FIELD — INCONSISTENT FORMATTING & SPELLING
# Problem: 35+ unique raw Status values found for what should
#          be ~6 canonical statuses. Issues include mixed case,
#          typos ('Deliverd', 'Cancled'), abbreviations ('dlvrd',
#          'OFD', 'Cncld'), and synonymous terms ('Complete' vs
#          'Delivered', 'Quoted' vs 'Quote').
# Fix: Map all variants to 6 canonical statuses using a
#      deterministic lookup function.
# ============================================================

def standardize_status(s):
    """
    Map raw Status values to 6 canonical standard statuses.
    Canonical set: Delivered, In Transit, Quote, Canceled, Complete, Dispatched
    """
    if pd.isna(s):
        return 'Unknown'
    s_lower = str(s).strip().lower()

    # Delivered variants
    if s_lower in ['delivered', 'deliverd', 'deliver', 'dlvrd']:
        return 'Delivered'
    # Complete / Completed → treated as terminal like Delivered
    elif s_lower in ['complete', 'completed']:
        return 'Complete'
    # In Transit variants
    elif s_lower in ['in transit', 'in-transit', 'intransit', 'in_transit']:
        return 'In Transit'
    # Out for Delivery → In Transit sub-state
    elif s_lower in ['out for delivery', 'ofd']:
        return 'Out for Delivery'
    # Dispatched
    elif s_lower == 'dispatched':
        return 'Dispatched'
    # Committed
    elif s_lower == 'committed':
        return 'Committed'
    # Quote variants (pre-shipment)
    elif s_lower in ['quote', 'quoted', 'ready', 'sent']:
        return 'Quote'
    # Canceled variants
    elif s_lower in ['canceled', 'cancelled', 'cncld', 'cancled',
                     'canceled', 'cancelled']:
        return 'Canceled'
    else:
        return 'Unknown'

df['Status'] = df['Status'].apply(standardize_status)
print(f"Issue 2 - Status standardized. Unique values now: {df['Status'].nunique()}")
print("  ", df['Status'].value_counts().to_dict())

# ============================================================
# ISSUE 3: TRAILER TYPE — SEVERE INCONSISTENCY (120+ variants)
# Problem: 120+ unique raw Trailer Type values exist for what
#          are ~10-12 canonical equipment types. Issues include:
#          - Mixed case: 'Van', 'VAN', 'van', ' Van'
#          - Abbreviations: 'P/O' for Power Only
#          - Verbose names: 'FiftyThreeReefer' for '53 ft Reefer'
#          - Descriptive suffixes: 'van (dry)', 'Van | Dry'
#          - Null values (2,415 rows)
#          - Generic 'Not Specified' / 'Unspecified'
# Fix: Keyword-based mapping to 12 canonical trailer types.
#      NULL and 'Not Specified' → 'Unknown'
# ============================================================

def standardize_trailer(s):
    """
    Map 120+ raw Trailer Type variants to 12 canonical types
    using keyword detection logic (case-insensitive).
    
    Canonical types:
      Van, Flatbed, LTL, Reefer, Power Only, Hotshot,
      Straight Truck, Sprinter, Step Deck, Conestoga,
      Intermodal, Unknown
    """
    if pd.isna(s):
        return 'Unknown'
    
    s_clean = str(s).strip().lower()
    
    # Reject placeholder values
    if s_clean in ['not specified', 'unspecified', 'other - see accessorials',
                   'n/a', '']:
        return 'Unknown'
    
    # Power Only — check BEFORE 'van' to avoid 'power van' mismatch
    if any(k in s_clean for k in ['power only', 'poweronly', 'power-only', 'p/o']):
        return 'Power Only'
    
    # Reefer — refrigerated trailer (check before Van since 'van reefer' exists)
    if any(k in s_clean for k in ['reefer', 'refrigerated', 'rf ']):
        return 'Reefer'
    
    # Flatbed variants (check before generic 'flat')
    if any(k in s_clean for k in ['flatbed', 'flat bed', 'flat-bed', 'flatb']):
        return 'Flatbed'
    
    # Conestoga (specialized flatbed cover)
    if 'conestoga' in s_clean or 'curtain side' in s_clean:
        return 'Conestoga'
    
    # Step Deck
    if any(k in s_clean for k in ['step deck', 'stepdeck']):
        return 'Step Deck'
    
    # Hotshot (small flatbed, typically <40ft)
    if any(k in s_clean for k in ['hotshot', 'hot shot', 'hot-shot', 'hotsht']):
        return 'Hotshot'
    
    # Straight Truck / Box Truck
    if any(k in s_clean for k in ['straight truck', 'straighttruck', 'straight-truck',
                                    'city truck', '12 ft', '24 ft', '26 ft']):
        return 'Straight Truck'
    
    # Sprinter Van (smaller than full straight truck)
    if 'sprinter' in s_clean:
        return 'Sprinter'
    
    # Intermodal / Container
    if any(k in s_clean for k in ['intermodal', 'container', 'ocean cont']):
        return 'Intermodal'
    
    # LTL (Less-Than-Truckload — equipment type in this context)
    if any(k in s_clean for k in ['ltl', 'less than', 'less-than', 'l.t.l.']):
        return 'LTL'
    
    # Van (dry van — most common, broad fallback for Van variants)
    if any(k in s_clean for k in ['van', 'dry van', 'dryvan']):
        return 'Van'
    
    # Air Freight
    if any(k in s_clean for k in ['air freight', 'dom. air', 'int. air']):
        return 'Air Freight'
    
    # Low Boy / RGN (specialized heavy equipment)
    if any(k in s_clean for k in ['low boy', 'lowboy', 'rgn', 'double drop']):
        return 'Low Boy / RGN'
    
    # Tanker
    if 'tanker' in s_clean or 'liquid bulk' in s_clean:
        return 'Tanker'
    
    return 'Unknown'

df['Trailer Type'] = df['Trailer Type'].apply(standardize_trailer)
print(f"Issue 3 - Trailer Type standardized. Unique values now: {df['Trailer Type'].nunique()}")
print("  ", df['Trailer Type'].value_counts().to_dict())

# ============================================================
# ISSUE 4: SHIPMENT TYPE — INCONSISTENT FORMATTING
# Problem: ~40 variants of 4 core types (Truckload, LTL,
#          Drayage, Hot Shot). Includes case inconsistency,
#          abbreviations (T/L, TL), and synonym variants.
# Fix: Keyword-based canonical mapping to 6 types.
# ============================================================

def standardize_shipment_type(s):
    """
    Normalize Shipment Type to: Truckload, LTL, Drayage,
    Hot Shot, Intermodal, Air Freight, Other
    """
    if pd.isna(s):
        return 'Unknown'
    s_clean = str(s).strip().lower()
    
    if s_clean in ['unspecified', 'not specified', 'n/a', '']:
        return 'Unknown'
    if any(k in s_clean for k in ['truckload', 'truck load', 'truck-load',
                                    't/l', ' tl', 'tl ']):
        return 'Truckload'
    elif s_clean in ['tl', 'tl']:
        return 'Truckload'
    elif any(k in s_clean for k in ['ltl', 'less than', 'less-than', 'l.t.l.']):
        return 'LTL'
    elif any(k in s_clean for k in ['drayage', 'dray']):
        return 'Drayage'
    elif any(k in s_clean for k in ['hot shot', 'hotshot', 'hot-shot']):
        return 'Hot Shot'
    elif 'intermodal' in s_clean:
        return 'Intermodal'
    elif any(k in s_clean for k in ['air freight', 'air', 'dom. air', 'int. air']):
        return 'Air Freight'
    elif any(k in s_clean for k in ['partial', 'rf tl', 'rf ltl',
                                     'rail', 'bulk', 'services']):
        return 'Other'
    return 'Unknown'

df['Shipment Type'] = df['Shipment Type'].apply(standardize_shipment_type)
print(f"Issue 4 - Shipment Type standardized. Unique values: {df['Shipment Type'].nunique()}")

# ============================================================
# ISSUE 5: MARGIN TOTAL INCONSISTENCY WITH BUY/SELL
# Problem: For 97,849 rows, Margin Total ≠ Sell Total - Buy Total.
#          Differences range from -$1,006 to +$1,145.
#          Investigation: The Margin Total column appears to
#          represent an ADJUSTED margin (after accessorials,
#          fuel surcharges, or other fees not reflected in
#          Buy/Sell Totals). The raw Sell-Buy delta is the
#          GROSS margin; Margin Total is the NET margin.
# Fix: Preserve Margin Total as the authoritative net figure.
#      Add a Gross Margin column for transparency.
#      Add a Margin % and flag for negative margin rows.
# ============================================================

# Preserve original Margin Total (net, as-recorded)
# Add Gross Margin for analytical reference
df['Gross Margin'] = (df['Sell Total'] - df['Buy Total']).round(2)

# Margin % = Net Margin / Sell Total (avoid div-by-zero)
df['Margin Pct'] = np.where(
    df['Sell Total'] != 0,
    (df['Margin Total'] / df['Sell Total'] * 100).round(2),
    np.nan
)

# Flag negative margin loads (key for dashboard filter requirement)
df['Negative Margin Flag'] = df['Margin Total'] < 0

print(f"Issue 5 - Margin columns added. Negative margin rows: {df['Negative Margin Flag'].sum():,}")

# ============================================================
# ISSUE 6: MISSING VALUES IN KEY FIELDS
# Problem: Multiple columns have significant nulls:
#   - Sales Rep Name: 32,412 nulls
#   - Carrier Rep: 53,313 nulls
#   - Origin/Destination ZIP, Country, Street: 14,000-28,000 nulls
#   - Trailer Type: 2,415 nulls (handled in Issue 3)
#   - Linehaul Carrier Name: 7,371 nulls
#   - Account Manager / SSR: 3,418 nulls each
# Fix: Fill with 'Unknown' / 'Not Assigned' placeholders for
#      categorical fields. Numerical nulls (ZIP, Mileage, Weight)
#      remain NULL — imputation would introduce false precision.
# ============================================================

# Categorical fill with placeholder (preserves unknown signal)
df['Sales Rep Name'] = df['Sales Rep Name'].fillna('Unassigned')
df['Carrier Rep'] = df['Carrier Rep'].fillna('Unassigned')
df['Account Manager'] = df['Account Manager'].fillna('Unassigned')
df['SSR'] = df['SSR'].fillna('Unassigned')
df['Linehaul Carrier Name'] = df['Linehaul Carrier Name'].fillna('Unknown Carrier')
df['Origin Country'] = df['Origin Country'].fillna('Unknown')
df['Destination Country'] = df['Destination Country'].fillna('Unknown')

# Address fields: fill with 'Not Provided' (for display, not join)
df['Origin Street Address'] = df['Origin Street Address'].fillna('Not Provided')
df['Destination Street Address'] = df['Destination Street Address'].fillna('Not Provided')
df['Destination Company Name'] = df['Destination Company Name'].fillna('Unknown')

print(f"Issue 6 - Null categorical fields filled with placeholders")

# ============================================================
# ISSUE 7: ZERO BUY/SELL TOTAL VALUES (data entry anomalies)
# Problem: 4,117 rows have $0 Buy Total; 2,805 rows have $0
#          Sell Total. These are operationally invalid — a
#          brokerage always has a cost and revenue figure.
#          These may be Quote-stage records or data entry errors.
# Fix: Flag these rows. Do not drop — they may be valid Quotes.
#      Tag them with a data quality warning for ops review.
# ============================================================

df['Zero_Financial_Flag'] = (df['Buy Total'] == 0) | (df['Sell Total'] == 0)
print(f"Issue 7 - Zero financial value flags added: {df['Zero_Financial_Flag'].sum():,} rows flagged")

# ============================================================
# ISSUE 8: SHIPMENT TYPE 'TL' EDGE CASE
# Already handled in Issue 4. Verifying standalone 'TL' rows.
# ============================================================
# Handled in standardize_shipment_type above.

# ============================================================
# STEP FINAL: SORT & RESET INDEX
# ============================================================
df.sort_values('Pickup Date', inplace=True)
df.reset_index(drop=True, inplace=True)

# ============================================================
# STEP FINAL: SAVE CLEANED DATASET
# ============================================================
output_path = path / 'output' / 'Freight_Cleaned_Dataset.xlsx'
df.to_excel(output_path, index=False)

print(f"\n{'='*60}")
print(f"CLEANING COMPLETE")
print(f"Final row count: {len(df):,}")
print(f"Final column count: {df.shape[1]}")
print(f"Saved to: {output_path}")
print(f"{'='*60}")

# Summary statistics for report
print("\n=== FINAL STATUS DISTRIBUTION ===")
print(df['Status'].value_counts())
print("\n=== FINAL TRAILER TYPE DISTRIBUTION ===")
print(df['Trailer Type'].value_counts())
print("\n=== FINAL SHIPMENT TYPE DISTRIBUTION ===")
print(df['Shipment Type'].value_counts())
