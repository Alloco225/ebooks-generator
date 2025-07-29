import { BUSINESS_IDEAS } from "./data/ideas";

console.log("📘 Business Idea Ebook Generator - Testing");
console.log("\n📂 Available Categories:");

const categories = Object.keys(BUSINESS_IDEAS);
categories.forEach((cat, i) => {
  console.log(`${i + 1}. ${cat}`);
});

console.log(`\n✅ Total categories: ${categories.length}`);

// Test one category
const firstCategory = categories[0];
const ideas = BUSINESS_IDEAS[firstCategory as keyof typeof BUSINESS_IDEAS];
console.log(`\n💡 Sample ideas from "${firstCategory}":`);
ideas.forEach((idea: string, i: number) => {
  console.log(`  ${i + 1}. ${idea}`);
});

console.log("\n🎉 Test completed successfully!"); 