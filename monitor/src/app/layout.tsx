import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VLA-Shield Monitor",
  description: "Real-time safety monitoring console for VLA-Shield runtime",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-surface text-gray-100 min-h-screen font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
