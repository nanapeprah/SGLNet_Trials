import "./components.css";
import TextField from "@mui/material/TextField";
import Popper from "./Popper";
import Button from "@mui/material/Button";
import { useNavigate } from "react-router-dom";

export default function Login() {
  const navigate = useNavigate();

  function redirect() {
    let password = document.getElementById("password")
      ? document.getElementById("password")
      : "love";
    if (password.value === "iloveyou") {
      navigate("/home");
    } else {
      console.log("Incorrect password");
    }
  }

  return (
    <div className="login-form">
      <h2 className="login-heading">Login</h2>
      <TextField variant="outlined" id="password" label="Password" />
      <div className="hint-box">
        <Popper />
        <Button variant="contained" onClick={redirect}>
          Submit
        </Button>
      </div>
    </div>
  );
}
