#!/bin/bash

rm -rfd tests/test_projects
# Project 1: Java
mkdir -p tests/test_projects/java_menu/src/main/java/com/example
cat > tests/test_projects/java_menu/src/main/java/com/example/MenuApp.java << 'EOF'
package com.example;

import java.sql.*;
import java.util.Scanner;

public class MenuApp {
    private static final String DB_URL = "jdbc:sqlite:menu.db";
    
    public static void main(String[] args) {
        try {
            // Initialize database
            initializeDatabase();
            
            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.println("\n=== CLI Menu ===");
                System.out.println("1. Add text");
                System.out.println("2. Exit");
                System.out.print("Choose an option: ");
                
                String choice = scanner.nextLine();
                
                switch (choice) {
                    case "1":
                        System.out.print("Enter text: ");
                        String input = scanner.nextLine();
                        System.out.println("You entered: " + input);
                        saveToDatabase(input);
                        break;
                    case "2":
                        System.out.println("Goodbye!");
                        scanner.close();
                        System.exit(0);
                    default:
                        System.out.println("Invalid option");
                }
            }
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }
    
    private static void initializeDatabase() throws SQLException {
        try (Connection conn = DriverManager.getConnection(DB_URL);
             Statement stmt = conn.createStatement()) {
            stmt.execute("CREATE TABLE IF NOT EXISTS entries (text VARCHAR(255))");
        }
    }
    
    private static void saveToDatabase(String text) throws SQLException {
        try (Connection conn = DriverManager.getConnection(DB_URL);
             PreparedStatement pstmt = conn.prepareStatement("INSERT INTO entries (text) VALUES (?)")) {
            pstmt.setString(1, text);
            pstmt.executeUpdate();
            System.out.println("Saved to database!");
        }
    }
}
EOF

# Project 2: Node.js
mkdir -p tests/test_projects/nodejs_menu
cat > tests/test_projects/nodejs_menu/index.js << 'EOF'
const readline = require('readline');
const sqlite3 = require('sqlite3').verbose();

const db = new sqlite3.Database('menu.db');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Initialize database
db.run(`CREATE TABLE IF NOT EXISTS entries (text TEXT)`, showMenu);

function showMenu() {
    console.log('\n=== CLI Menu ===');
    console.log('1. Add text');
    console.log('2. Exit');
    rl.question('Choose an option: ', handleChoice);
}

function handleChoice(choice) {
    switch (choice) {
        case '1':
            rl.question('Enter text: ', (input) => {
                console.log('You entered:', input);
                db.run('INSERT INTO entries (text) VALUES (?)', [input], (err) => {
                    if (err) console.error(err);
                    else console.log('Saved to database!');
                    showMenu();
                });
            });
            break;
        case '2':
            console.log('Goodbye!');
            db.close();
            rl.close();
            break;
        default:
            console.log('Invalid option');
            showMenu();
    }
}
EOF
cat > tests/test_projects/nodejs_menu/package.json << 'EOF'
{
    "name": "nodejs_menu",
    "version": "1.0.0",
    "main": "index.js",
    "dependencies": {
        "sqlite3": "^5.1.7"
    }
}
EOF

# Project 3: Scala
mkdir -p tests/test_projects/scala_menu/src/main/scala
cat > tests/test_projects/scala_menu/src/main/scala/MenuApp.scala << 'EOF'
import java.sql.{Connection, DriverManager, PreparedStatement}
import scala.io.StdIn

object MenuApp {
  val DB_URL = "jdbc:sqlite:menu.db"
  
  def main(args: Array[String]): Unit = {
    initializeDatabase()
    while (true) {
      println("\n=== CLI Menu ===")
      println("1. Add text")
      println("2. Exit")
      print("Choose an option: ")
      
      StdIn.readLine() match {
        case "1" =>
          print("Enter text: ")
          val input = StdIn.readLine()
          println(s"You entered: $input")
          saveToDatabase(input)
        case "2" =>
          println("Goodbye!")
          System.exit(0)
        case _ =>
          println("Invalid option")
      }
    }
  }
  
  def initializeDatabase(): Unit = {
    val conn = DriverManager.getConnection(DB_URL)
    val stmt = conn.createStatement()
    try {
      stmt.execute("CREATE TABLE IF NOT EXISTS entries (text VARCHAR(255))")
    } finally {
      stmt.close()
      conn.close()
    }
  }
  
  def saveToDatabase(text: String): Unit = {
    val conn = DriverManager.getConnection(DB_URL)
    val pstmt = conn.prepareStatement("INSERT INTO entries (text) VALUES (?)")
    try {
      pstmt.setString(1, text)
      pstmt.executeUpdate()
      println("Saved to database!")
    } finally {
      pstmt.close()
      conn.close()
    }
  }
}
EOF

# Project 4: Python
mkdir -p tests/test_projects/python_menu
cat > tests/test_projects/python_menu/main.py << 'EOF'
import sqlite3
import sys

def initialize_database():
    conn = sqlite3.connect('menu.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS entries (text TEXT)')
    conn.commit()
    conn.close()

def save_to_database(text):
    conn = sqlite3.connect('menu.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO entries (text) VALUES (?)', (text,))
    conn.commit()
    conn.close()
    print("Saved to database!")

def show_menu():
    while True:
        print("\n=== CLI Menu ===")
        print("1. Add text")
        print("2. Exit")
        choice = input("Choose an option: ")
        
        if choice == "1":
            text = input("Enter text: ")
            print(f"You entered: {text}")
            save_to_database(text)
        elif choice == "2":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option")

if __name__ == "__main__":
    initialize_database()
    show_menu()
EOF