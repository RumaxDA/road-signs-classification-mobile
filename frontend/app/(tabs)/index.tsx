import React from "react";
import { StyleSheet, Text, ImageBackground, View } from "react-native";

export default function HomeScreen() {
  return (
    <ImageBackground
      source={require("../../assets/images/bg.jpg")}
      style={styles.background}
      resizeMode="cover"
    >
      <View style={styles.inner}>
        <Text style={styles.text}>Tło działa!</Text>
      </View>
    </ImageBackground>
  );
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    width: "100%",
    height: "100%",
  },
  inner: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  text: {
    color: "white",
    fontSize: 24,
  },
});
