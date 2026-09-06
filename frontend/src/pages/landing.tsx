import "../../stylesheets/landing.css";

const welcome_messages = [
  "What's on your mind today?",
  "Ready when you are.",
  "Ready to dive in?",
  "Where should we begin?",
  "Good to see you!",
  "How can I help?",
];

const random_welcome_message =
  welcome_messages[Math.floor(Math.random() * welcome_messages.length)];

export const Landing = () => {
  return (
    <div className="landing-page">
      <div className="welcome-message">{random_welcome_message}</div>
      <div className="message-bar">
        <form className="bar-wrapper">
          <input className="text-input" type="text"></input>
          <button className="send-button" type="submit">
            (↑)
          </button>
        </form>
      </div>
    </div>
  );
};
