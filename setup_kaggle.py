"""
Kaggle API Setup Script for Flower Classification Model
Run this script to configure your Kaggle API credentials
"""

import os
import json
import shutil
from pathlib import Path

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    
    print("=" * 60)
    print("KAGGLE API SETUP")
    print("=" * 60)
    print()
    
    # Check common locations for kaggle.json
    home_dir = Path.home()
    possible_locations = [
        home_dir / '.kaggle' / 'kaggle.json',
        Path.cwd() / 'kaggle.json',
        Path(os.environ.get('KAGGLE_CONFIG_DIR', '')) / 'kaggle.json' if os.environ.get('KAGGLE_CONFIG_DIR') else None,
    ]
    
    # Filter out None values
    possible_locations = [loc for loc in possible_locations if loc]
    
    # Check if kaggle.json exists in any location
    kaggle_json_found = False
    for location in possible_locations:
        if location.exists():
            print(f"✓ Found kaggle.json at: {location}")
            kaggle_json_found = True
            break
    
    if not kaggle_json_found:
        print("❌ kaggle.json not found!")
        print()
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click on 'API' in the left sidebar")
        print("3. Click 'Create New Token'")
        print("4. Download the kaggle.json file")
        print()
        print("Then copy the kaggle.json file to ONE of these locations:")
        print(f"   - {home_dir / '.kaggle' / 'kaggle.json'}")
        print(f"   - {Path.cwd() / 'kaggle.json'}")
        print()
        
        # Ask user where they put it
        print("Where did you place the kaggle.json file?")
        print("1. In project folder (kaggle.json)")
        print("2. In home folder (.kaggle/kaggle.json)")
        print("3. Other location")
        print()
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
            kaggle_path = Path.cwd() / 'kaggle.json'
        elif choice == '2':
            kaggle_path = home_dir / '.kaggle' / 'kaggle.json'
        else:
            kaggle_path = input("Enter full path to kaggle.json: ").strip()
            kaggle_path = Path(kaggle_path)
        
        if not kaggle_path.exists():
            print(f"❌ File not found at: {kaggle_path}")
            return False
    
    # Set proper permissions (Windows doesn't need chmod but we'll set env var)
    os.environ['KAGGLE_CONFIG_DIR'] = str(home_dir / '.kaggle')
    
    # Test Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authenticated successfully!")
        print()
        
        # Test dataset access
        print("Testing dataset access...")
        dataset_list = api.dataset_list(user='iamaditey', search='flower')
        if dataset_list:
            print(f"✓ Found flower dataset: {dataset_list[0].ref}")
            return True
        else:
            print("⚠ Could not find flower dataset")
            return False
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure kaggle.json is valid JSON")
        print("2. Check that your Kaggle account is active")
        print("3. Try regenerating the API token")
        return False

if __name__ == '__main__':
    success = setup_kaggle_api()
    if success:
        print()
        print("=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
        print()
        print("You can now run: python train.py")
        print("This will download the flower dataset and train the model.")
    else:
        print()
        print("=" * 60)
        print("SETUP FAILED")
        print("=" * 60)
        print()
        print("Please complete the setup steps above, then try again.")
