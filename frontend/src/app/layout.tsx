import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const flightSans = Space_Grotesk({
  variable: "--font-flight-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const flightMono = JetBrains_Mono({
  variable: "--font-flight-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Flight Delay Prediction Console",
  description:
    "Air-traffic inspired chat console that connects to your delay prediction endpoint.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${flightSans.variable} ${flightMono.variable} antialiased bg-slate-50 text-slate-900`}
      >
        {children}
      </body>
    </html>
  );
}
