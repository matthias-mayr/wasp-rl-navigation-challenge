#!/bin/bash 

echo -e "Available Java versions:\n"
sudo  update-java-alternatives --list

echo -e "\nChoosing Java 8"
echo -e "WARNING: You are responsible to return to your old configuration!\n"
sudo  update-java-alternatives --set java-1.8.0-openjdk-amd64