import july from "../Assets/july.jpeg";
import august from "../Assets/august.jpeg";
import sept from "../Assets/sept.jpeg";
import oct from "../Assets/oct.jpeg";
import nov from "../Assets/nov.jpeg";
import dec from "../Assets/dec.jpeg";
import jan from "../Assets/jan.jpeg";
import march from "../Assets/march.jpeg";

export default function Gallery() {
  return (
    <div className="gallery">
      <p>Gallery</p>
      <img src={july} className="images" width="80%" alt="july" />
      <p>July</p>
      <img src={august} className="images" width="80%" alt="august" />
      <p>August</p>
      <img src={sept} className="images" width="80%" alt="sept" />
      <p>September</p>
      <img src={oct} className="images" width="80%" alt="oct" />
      <p>October</p>
      <img src={nov} className="images" width="80%" alt="nov" />
      <p>November</p>
      <img src={dec} className="images" width="80%" alt="dec" />
      <p>December</p>
      <img src={jan} className="images" width="80%" alt="dec" />
      <p>January</p>
      <img src={march} className="images" width="80%" alt="dec" />
      <p>March</p>
    </div>
  );
}
