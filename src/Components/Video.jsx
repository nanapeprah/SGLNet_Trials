import video from "../Assets/pytorch.mp4";

export default function Video() {
  return (
    <div className="video">
      <p>Video</p>
      <video width="100%" height="100%" autoPlay>
        <source src={video} />
      </video>
    </div>
  );
}
