"""
Contact Manager - Starter Template
Session 4: Functions and Data Structures
Anna Smirnova, October 2025

This is your starting point! Fill in the TODOs to build a working contact manager.
Try to implement these yourself before looking at solutions.
"""

import json

# Initialize empty contacts list
contacts = []

# ============================================================================
# FEATURE 1: ADD CONTACTS
# ============================================================================

def add_contact(contacts_list, name, phone, email=""):
    """Add a new contact to our list"""
    # TODO: Create a dictionary for this contact with:
    # - "name": name
    # - "phone": phone
    # - "email": email
    # - "id": len(contacts_list) + 1

    # TODO: Add the contact dictionary to contacts_list using .append()

    # TODO: Print a success message like "✓ Added {name} to contacts"

    # TODO: Return the contact dictionary
    pass


def contact_exists(contacts_list, name):
    """Check if a contact already exists"""
    # TODO: Loop through contacts_list
    # TODO: For each contact, check if contact["name"].lower() == name.lower()
    # TODO: If match found, return True
    # TODO: If no match found after loop, return False
    pass


def add_contact_safe(contacts_list, name, phone, email=""):
    """Add contact with duplicate prevention"""
    # TODO: Use contact_exists() to check if name already exists
    # TODO: If exists, print warning message and return None
    # TODO: Otherwise, call add_contact() and return its result
    pass


# ============================================================================
# FEATURE 2: SEARCH CONTACTS
# ============================================================================

def search_contacts(contacts_list, search_term):
    """Find contacts by name or phone"""
    # TODO: Create empty list called results
    # TODO: Convert search_term to lowercase for case-insensitive search

    # TODO: Loop through contacts_list
    # TODO: Check if search_term is in contact["name"] OR contact["phone"]
    # TODO: If match, add contact to results

    # TODO: Return results list
    pass


def display_search_results(search_term):
    """Display search results in a user-friendly way"""
    # TODO: Call search_contacts() to get results

    # TODO: If no results, print "No contacts found for '{search_term}'"
    # TODO: Otherwise, print "Found X contact(s):"
    # TODO: For each result, print "  • {name}: {phone}"
    pass


# ============================================================================
# FEATURE 3: DISPLAY ALL CONTACTS
# ============================================================================

def display_all_contacts(contacts_list):
    """Display all contacts in a formatted way"""
    # TODO: Check if contacts_list is empty
    # TODO: If empty, print "No contacts to display" and return

    # TODO: Print a header line like "ID  Name        Phone       Email"
    # TODO: Loop through contacts_list
    # TODO: For each contact, print formatted line with all fields
    pass


# ============================================================================
# FEATURE 4: SAVE/LOAD
# ============================================================================

def save_contacts(contacts_list, filename="contacts.json"):
    """Save contacts to a file"""
    try:
        # TODO: Open file in write mode using 'with open(filename, 'w') as f:'
        # TODO: Use json.dump(contacts_list, f, indent=2) to save
        # TODO: Print success message
        # TODO: Return True
        pass
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False


def load_contacts(filename="contacts.json"):
    """Load contacts from a file"""
    try:
        # TODO: Open file in read mode using 'with open(filename, 'r') as f:'
        # TODO: Use json.load(f) to read contacts
        # TODO: Print success message
        # TODO: Return the loaded contacts
        pass
    except FileNotFoundError:
        print("No saved contacts found")
        return []
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return []


# ============================================================================
# FEATURE 5: DELETE CONTACT
# ============================================================================

def delete_contact(contacts_list, name):
    """Delete a contact by name"""
    # TODO: Loop through contacts_list with enumerate() to get index and contact
    # TODO: If contact["name"].lower() == name.lower():
    #       - Use contacts_list.pop(i) to remove it
    #       - Print success message
    #       - Return the removed contact
    # TODO: If not found, print "Contact not found" and return None
    pass


# ============================================================================
# FEATURE 6: STATISTICS
# ============================================================================

def get_contact_stats(contacts_list):
    """Get interesting statistics about contacts"""
    # TODO: Create a dictionary called stats with:
    #       - "total": len(contacts_list)
    #       - "with_email": 0
    #       - "without_email": 0

    # TODO: Loop through contacts_list
    # TODO: For each contact, check if contact["email"] is not empty
    #       - If yes, increment stats["with_email"]
    #       - If no, increment stats["without_email"]

    # TODO: Return stats dictionary
    pass


# ============================================================================
# MENU SYSTEM
# ============================================================================

def display_menu():
    """Display the main menu"""
    print("\n" + "="*40)
    print("     CONTACT MANAGER MENU")
    print("="*40)
    print("1. Add contact")
    print("2. Search contacts")
    print("3. Display all contacts")
    print("4. Show statistics")
    print("5. Save contacts")
    print("6. Load contacts")
    print("7. Delete contact")
    print("0. Exit")
    print("="*40)


def run_contact_manager():
    """Main program loop"""
    # TODO: Call load_contacts() and store result in local_contacts

    # TODO: Create infinite loop with 'while True:'
    # TODO: Display menu
    # TODO: Get user choice with input()

    # TODO: If choice is "0":
    #       - Save contacts
    #       - Print goodbye message
    #       - Break the loop

    # TODO: If choice is "1":
    #       - Get name, phone, email from user
    #       - Call add_contact_safe()

    # TODO: If choice is "2":
    #       - Get search term from user
    #       - Call display_search_results()

    # TODO: If choice is "3":
    #       - Call display_all_contacts()

    # TODO: If choice is "4":
    #       - Call get_contact_stats()
    #       - Print the statistics

    # TODO: If choice is "5":
    #       - Call save_contacts()

    # TODO: If choice is "6":
    #       - Call load_contacts() and update local_contacts

    # TODO: If choice is "7":
    #       - Get name from user
    #       - Call delete_contact()

    # TODO: For any other choice, print "Invalid choice"
    pass


# ============================================================================
# BONUS: RECURSION EXAMPLE
# ============================================================================

def find_contact_recursive(contacts_list, name, index=0):
    """Find a contact using recursion (just for learning!)"""
    # TODO: Base case 1 - if index >= len(contacts_list), return None

    # TODO: Base case 2 - if contacts_list[index]["name"].lower() == name.lower():
    #       return contacts_list[index]

    # TODO: Recursive case - return find_contact_recursive(contacts_list, name, index + 1)
    pass


# ============================================================================
# TEST YOUR CODE
# ============================================================================

if __name__ == "__main__":
    print("Contact Manager v1.0 - Starter Template")
    print("=" * 50)

    # Test your functions here!
    # Example:
    # add_contact_safe(contacts, "Alice Smith", "555-0001", "alice@email.com")
    # display_all_contacts(contacts)

    # Uncomment to run the full menu system:
    # run_contact_manager()

    print("\n💡 Tip: Uncomment the test lines above to try your code!")
    print("💡 Tip: Implement one function at a time and test it!")
