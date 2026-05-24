import type { ChatMessage } from "@infinite-edge-agent/core";

interface ChatMessageViewProps {
  message: ChatMessage;
}

export function ChatMessageView({ message }: ChatMessageViewProps) {
  return (
    <article className={`message message-${message.role}`}>
      <header>
        <strong>{message.role}</strong>
        <time>{new Date(message.createdAt).toLocaleTimeString()}</time>
      </header>
      <p>{message.content}</p>
    </article>
  );
}
