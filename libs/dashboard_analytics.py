import re
import logging
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from datetime import datetime
import calendar

logger = logging.getLogger(__name__)

class TransactionAnalyzer:
    """Analyzes transactions for dashboard insights"""
    
    def __init__(self):
        # Category keywords for classification
        self.category_keywords = {
            # Spending Categories
            'Food & Dining': [
                'swiggy', 'zomato', 'restaurant', 'cafe', 'food', 'dominos', 'pizza', 
                'kfc', 'mcdonalds', 'burger', 'hotel', 'dining', 'meal', 'kitchen',
                'grocery', 'supermarket', 'bigbasket', 'grofers', 'reliance fresh'
            ],
            'Transportation': [
                'uber', 'ola', 'rapido', 'metro', 'bus', 'taxi', 'auto', 'petrol', 'fuel',
                'diesel', 'parking', 'toll', 'transport', 'railway', 'flight', 'cab'
            ],
            'Shopping': [
                'amazon', 'flipkart', 'myntra', 'ajio', 'shopping', 'purchase', 'retail',
                'store', 'mall', 'buy', 'order', 'delivery', 'ecommerce'
            ],
            'Bills & Utilities': [
                'electricity', 'water', 'gas', 'mobile', 'internet', 'wifi', 'recharge',
                'bill', 'utility', 'phone', 'broadband', 'airtel', 'jio', 'vi', 'bsnl'
            ],
            'Healthcare': [
                'hospital', 'doctor', 'medical', 'pharmacy', 'medicine', 'health',
                'clinic', 'diagnostic', 'lab', 'test', 'checkup'
            ],
            'Entertainment': [
                'netflix', 'amazon prime', 'hotstar', 'movie', 'cinema', 'entertainment',
                'gaming', 'spotify', 'music', 'subscription', 'ott'
            ],
            'Education': [
                'school', 'college', 'university', 'course', 'training', 'education',
                'fee', 'tuition', 'books', 'learning'
            ],
            'Financial Services': [
                'loan', 'emi', 'insurance', 'premium', 'bank', 'charges', 'fees',
                'interest', 'credit card', 'mutual fund'
            ],
            
            # Income Categories  
            'Salary': [
                'salary', 'payroll', 'wages', 'payment', 'income', 'monthly pay',
                'company', 'employer', 'organization'
            ],
            'Business Income': [
                'business', 'freelance', 'consulting', 'professional', 'services',
                'client payment', 'invoice', 'contract'
            ],
            'Investment Returns': [
                'dividend', 'interest credit', 'fd maturity', 'mutual fund',
                'stock', 'share', 'bond', 'returns', 'profit'
            ],
            'Refunds': [
                'refund', 'cashback', 'reward', 'points redemption', 'reversal'
            ],
            
            # Investment & Savings Categories
            'Mutual Funds': [
                'sip', 'mutual fund', 'mf', 'amc', 'systematic', 'investment plan',
                'equity', 'debt fund', 'elss'
            ],
            'Stock Market': [
                'zerodha', 'groww', 'upstox', 'angel broking', 'trading', 'demat',
                'stocks', 'shares', 'equity purchase'
            ],
            'Fixed Deposits': [
                'fd', 'fixed deposit', 'term deposit', 'recurring deposit', 'rd'
            ],
            'Savings Transfer': [
                'savings account', 'transfer to savings', 'internal transfer'
            ],
            'Insurance': [
                'lic', 'life insurance', 'term insurance', 'health insurance',
                'policy premium'
            ]
        }
        
        # Color scheme for categories
        self.category_colors = {
            'Food & Dining': '#FF6384',
            'Transportation': '#36A2EB', 
            'Shopping': '#FFCE56',
            'Bills & Utilities': '#4BC0C0',
            'Healthcare': '#9966FF',
            'Entertainment': '#FF9F40',
            'Education': '#FF6384',
            'Financial Services': '#C9CBCF',
            'Salary': '#4BC0C0',
            'Business Income': '#36A2EB',
            'Investment Returns': '#FFCE56',
            'Refunds': '#9966FF',
            'Mutual Funds': '#FF6384',
            'Stock Market': '#36A2EB',
            'Fixed Deposits': '#FFCE56',
            'Savings Transfer': '#4BC0C0',
            'Insurance': '#9966FF',
            'Other': '#E7E9ED'
        }

    def classify_transaction(self, description: str, amount_str: str, transaction_type: str = '') -> str:
        """Classify a transaction into a category based on description"""
        description_lower = description.lower()
        
        # First check for specific keywords
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return category
        
        # Fallback classification based on transaction type and patterns
        if transaction_type == 'credit':
            # Try to classify credits
            if any(word in description_lower for word in ['salary', 'pay', 'income']):
                return 'Salary'
            elif any(word in description_lower for word in ['transfer', 'neft', 'imps', 'upi']):
                return 'Business Income'
            else:
                return 'Other Income'
        else:
            # Try to classify debits
            if any(word in description_lower for word in ['transfer', 'neft', 'imps', 'upi']):
                return 'Other Transfer'
            elif any(word in description_lower for word in ['atm', 'withdrawal', 'cash']):
                return 'Cash Withdrawal'
            else:
                return 'Other Expense'
    
    def parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float"""
        if not amount_str or amount_str == '':
            return 0.0
        
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[₹$,\s]', '', str(amount_str))
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return 0.0

    def analyze_transactions(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze transactions and return dashboard data"""
        
        # Process each transaction
        processed_transactions = []
        spending_categories = defaultdict(float)
        income_categories = defaultdict(float) 
        investment_categories = defaultdict(float)
        monthly_data = defaultdict(lambda: {'income': 0.0, 'expenses': 0.0})
        
        for trans in transactions:
            # Extract and clean data
            description = trans.get('description', '')
            amount_str = trans.get('amount', '0')
            transaction_type = trans.get('transaction_type', '')
            date_str = trans.get('date', '')
            
            # Parse amount
            amount_numeric = self.parse_amount(amount_str)
            
            # Classify transaction
            category = self.classify_transaction(description, amount_str, transaction_type)
            
            # Add processed fields
            processed_trans = {
                **trans,
                'category': category,
                'amount_numeric': amount_numeric,
                'amount_str': amount_str
            }
            processed_transactions.append(processed_trans)
            
            # Categorize for dashboard
            if transaction_type == 'credit' or amount_numeric > 0:
                if category in ['Mutual Funds', 'Stock Market', 'Fixed Deposits', 'Savings Transfer', 'Insurance']:
                    investment_categories[category] += amount_numeric
                else:
                    income_categories[category] += amount_numeric
                
                # Monthly tracking
                month_key = self.extract_month(date_str)
                monthly_data[month_key]['income'] += amount_numeric
                    
            else:
                if category in ['Mutual Funds', 'Stock Market', 'Fixed Deposits', 'Savings Transfer', 'Insurance']:
                    investment_categories[category] += amount_numeric
                else:
                    spending_categories[category] += amount_numeric
                
                # Monthly tracking  
                month_key = self.extract_month(date_str)
                monthly_data[month_key]['expenses'] += amount_numeric

        # Calculate totals
        total_spending = sum(spending_categories.values())
        total_income = sum(income_categories.values())
        total_investments = sum(investment_categories.values())
        net_savings = total_income - total_spending - total_investments
        
        # Prepare chart data
        category_data = self.prepare_category_chart_data(spending_categories)
        monthly_chart_data = self.prepare_monthly_chart_data(monthly_data)
        
        return {
            'total_spending': total_spending,
            'total_income': total_income,
            'total_investments': total_investments,
            'net_savings': net_savings,
            'spending_categories': dict(sorted(spending_categories.items(), key=lambda x: x[1], reverse=True)),
            'income_categories': dict(sorted(income_categories.items(), key=lambda x: x[1], reverse=True)),
            'investment_categories': dict(sorted(investment_categories.items(), key=lambda x: x[1], reverse=True)),
            'category_labels': category_data['labels'],
            'category_amounts': category_data['amounts'],
            'category_colors': self.category_colors,
            'category_colors_array': category_data['colors'],
            'monthly_labels': monthly_chart_data['labels'],
            'monthly_income': monthly_chart_data['income'],
            'monthly_expenses': monthly_chart_data['expenses'],
            'recent_transactions': processed_transactions[-20:] if processed_transactions else []
        }
    
    def extract_month(self, date_str: str) -> str:
        """Extract month-year from date string"""
        try:
            # Try different date formats
            for fmt in ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%y', '%d/%m/%y']:
                try:
                    date_obj = datetime.strptime(date_str.strip(), fmt)
                    return f"{calendar.month_abbr[date_obj.month]} {date_obj.year}"
                except ValueError:
                    continue
            return "Unknown"
        except:
            return "Unknown"
    
    def prepare_category_chart_data(self, categories: Dict[str, float]) -> Dict[str, List]:
        """Prepare data for category pie chart"""
        if not categories:
            return {'labels': [], 'amounts': [], 'colors': []}
            
        # Sort by amount and take top 8 categories
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:8]
        
        labels = []
        amounts = []
        colors = []
        
        for category, amount in sorted_categories:
            labels.append(category)
            amounts.append(amount)
            colors.append(self.category_colors.get(category, '#E7E9ED'))
        
        return {
            'labels': labels,
            'amounts': amounts,
            'colors': colors
        }
    
    def prepare_monthly_chart_data(self, monthly_data: Dict[str, Dict]) -> Dict[str, List]:
        """Prepare data for monthly trend chart"""
        if not monthly_data:
            return {'labels': [], 'income': [], 'expenses': []}
        
        # Sort months chronologically
        sorted_months = sorted(monthly_data.keys())
        
        labels = []
        income = []
        expenses = []
        
        for month in sorted_months[-6:]:  # Last 6 months
            labels.append(month)
            income.append(monthly_data[month]['income'])
            expenses.append(monthly_data[month]['expenses'])
        
        return {
            'labels': labels,
            'income': income,
            'expenses': expenses
        }