import "./styles.css";
import * as React from "react";
import Box from "@mui/material/Box";
import BottomNavigation from "@mui/material/BottomNavigation";
import BottomNavigationAction from "@mui/material/BottomNavigationAction";
import CollectionsIcon from "@mui/icons-material/Collections";
import VideocamIcon from "@mui/icons-material/Videocam";
import DescriptionIcon from "@mui/icons-material/Description";
import Letter from "./Components/Letter";
import Video from "./Components/Video";
import Gallery from "./Components/Gallery";
import { BrowserRouter as Router } from "react-router-dom";
import { Route, Routes } from "react-router";
import Login from "./Components/Login";

export default function App() {
  const [value, setValue] = React.useState(0);

  class Screen extends React.Component {
    render() {
      if (value === 0) {
        return <Video />;
      }
      if (value === 1) {
        return <Gallery />;
      }
      if (value === 2) {
        return <Letter />;
      }
    }
  }

  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <>
              <Login />
            </>
          }
        />
        <Route
          path="/home"
          element={
            <>
              <Screen />
              <Box>
                <BottomNavigation
                  showLabels
                  value={value}
                  onChange={(event, newValue) => {
                    console.log(newValue);
                    setValue(newValue);
                  }}
                >
                  <BottomNavigationAction
                    label="Video"
                    icon={<VideocamIcon />}
                  />
                  <BottomNavigationAction
                    label="Gallery"
                    icon={<CollectionsIcon />}
                  />
                  <BottomNavigationAction
                    label="Letter"
                    icon={<DescriptionIcon />}
                  />
                </BottomNavigation>
              </Box>
            </>
          }
        />
      </Routes>
    </Router>
  );
}
